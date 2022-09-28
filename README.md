# ProVe's Pipeline Execution

ProVe is a pipelined tool for automated fact checking (AFC) on knowledge graphs (KGs), and is described in the paper "ProVe: A Pipeline for Automated Provenance Verification of Knowledge Graphs against Textual Sources".

This notebook organises the data, code, and models that make up ProVe. It also executes its different modules on WTR, an evaluation dataset consisting of KG triples and references extracted from Wikidata.

This repository has as main funcionatilites the following:
- Gathering and documenting the different modules/models that make up ProVe;
- Detailing the execution of ProVe on data extracted from Wikidata.

It consists of the following directories with their *sub-directories*/files:

## support_classification_pipeline_test.ipynb

This ipython notebook contains the code which runs the ProVe pipeline on WTR, a set of KG triples and references sampled from Wikidata.

## data
This folder contains scripts that parse Wikidata, as well as data and scripts that compile the KG triples and references sampled from Wikidata, originally found disjoined in the 'text_extraction' directory, into an unannotated version of WTR (annotation is carried out in another Repo, alongside evaluation). It contains:

- Wikidata_Claims_Parser.ipynb: A helper notebook which parses the triple and reference data from Wikidata
- wikidata_parser.py: A script with helper functions to parse Wikidata
- WTR_non_filtered_non_annotated.zip: A version of WTR obtained by joining the triple and reference data into a single dataset, but not yet annotated by the crowd
- Compiling_WTR_Non_Annotated.ipynb: A notebook which takes the triple and reference CSV files available in the directory 'text_extraction' and joins them into a single non-annotated dataset.

## text_extraction
This directory holds the KG triple (claims) and reference data sampled from Wikidata which constitute the evaluation dataset WTR. These references have already been resolved to URLs and had their text parsed and extracted with sliding windows of size 1 and 2 (as described in the ProVe paper). This process was carried out at a [different repository](https://github.com/gabrielmaia7/WD_Textual_References_Dataset). More info can be found in this directory's README. It contains:

- reference_html_as_sentences_df.csv: This file holds the 416 references samples from Wikidata, which represent WTR. For each reference, it shows their HTML code, the text retrieved from them, and its segmentation into multiple text segments which are then concatenated with sliding windows of size 1 and 2.
- text_reference_claims_df.csv: This file holds all the claims (KG triples) extracted for each of the 416 references in 'reference_html_as_sentences_df.csv'.

## verbalisation
This directory holds the scripts, by-product data, and models that consist of ProVe's verbalisation module. It contais:

- T5-Verbalisation-Test.ipynb: This notebook tests the verbalisation module found in the *graph2text* directory, ensuring it works properly.
- verbalisation_module.py: This script encapsulates handy implementation to call the verbalisation model in *graph2text*.
- verbalised_claims_df.csv: This dataset has, for each KG triple extracted in relation to one of the 416 references sampled from Wikidata, its component data, identifiers, and verbalisation as outputted by the model.
- verbalisation_claims_df_main.csv: This is 'verbalised_claims_df.csv' after main verbalisations are chosen, which is to say one verbalisation per unique reference-triple pair with the preferred labels for each triple component.
- verbalised_claims_df_main_corrected.csv: This is 'verbalised_claims_df_main.csv' after manual correction of those verbalisations that could not be improved by selecting alternative labels.
- verbalised_claims_df_final.csv: This is 'verbalised_claims_df_main_corrected.csv' after ontological triples and duplicated verbalisation-url pairs were removed. Then, an unique triple for each reference is selected, resulting in the 409 triples-reference pairs that make up WTR, since 7 are dropped due to having duplicated verbalisation-url combinations.
- *graph2text*: This repository holds the trained data verbalisation model. It is a slightly modified version of the [graph2text](https://github.com/UKPLab/plms-graph2text/) scripts by the folks at UKPLab, more specifically Ribeiro et al. We trained the model on webnlg as instructed in their repository, carrying here only the trained model and the code needed to run it. We have updated small bits of code as needed in order to adapt to more recent versions of pytorch and pytorch lightning. Some files were too big and were put on a cloud:
  - verbalisation/graph2text/outputs/t5-base_13881/val_avg_bleu=68.1000-step_count=5.ckpt: This file is not found in this repo and is instead found [here](https://emckclac-my.sharepoint.com/:u:/g/personal/k20036346_kcl_ac_uk/EbL1yTauXtpEqs4Izc97WNIBhumczrDGTNQb47uYGzXqsg?e=I9B5pR).
  - verbalisation/graph2text/outputs/t5-base_13881/best_tfmr/pytorch_model.bin: This file is not found in this repo and is instead found [here](https://emckclac-my.sharepoint.com/:u:/g/personal/k20036346_kcl_ac_uk/ES1YcFbwwIVJqpz0OcSnUsUBLvRnlV64AxtMiU8cczn9pw?e=yR56F3).

## sentence_retrieval
This directory holds the scripts, by-product data, and models that consist of ProVe's sentence selection/retrieval module. Some of the files here are slightly modified versions of KGAT's sentence retrieval model, by Liu et al., which can be found [here](https://github.com/thunlp/KernelGAT/tree/b6bba3a62aef42a2d0058b4c6dede690f97fc4db). We trained their [retrieval model](https://github.com/thunlp/KernelGAT/tree/b6bba3a62aef42a2d0058b4c6dede690f97fc4db/retrieval_model) exactly as described by their documentation. We then copied all files necessary to run the sentence retrieval model into this directory, updated them to newer versions of pytorch, and built an interface to access the model's functionality.

This directory contais:

- *bert_base*: This directory includes files that make up BERT base. The file 'sentence_retrieval/bert_base/pytorch_model.bin' was too big for GitHub and can be otherwise found [here](https://emckclac-my.sharepoint.com/:u:/g/personal/k20036346_kcl_ac_uk/EZVfly2hCntLvO3pmyJkesoBVrjlCbkUmJlVHPIzbdqwvA?e=KNWeBg).
- *checkpoint*: This directory is intended to hold the model checkpoints. These were too big for GitHub and can be found in:
  - sentence_retrieval/checkpoint/model.best.32.pt: This model is a newly trained sentence retrieval model with batch size 32 for training, the best performing out of the fresh models we trained, and can be found [here](https://emckclac-my.sharepoint.com/:u:/g/personal/k20036346_kcl_ac_uk/EVreut4deutKkTrPxA9a3PQBe2zccNCvsuuXn28nOm3n1w?e=4HfEIj).
  - sentence_retrieval/checkpoint/model.best.bkp.pt: This model is the one made available by Liu et al. at KGAT's original repository, and was used for benchmarking. It can be found [here](https://emckclac-my.sharepoint.com/:u:/g/personal/k20036346_kcl_ac_uk/EVreut4deutKkTrPxA9a3PQBe2zccNCvsuuXn28nOm3n1w?e=4HfEIj).
- bert_model.py: This code implements BERT base.
- data_loader.py: This code implements the functionality to feed the model with data for training/testing.
- file_utils.py: This code implements some utility functions for dealing with files.
- sentence_relevance_df.json: This is the resulting dataframe obtained by running the 409 triple-reference pairs through sentence selection. 
- sentence_retrieval_model.py: This code implements the sentence retrieval model by adding a linear + tanh head to BERT.
- sentence_retrieval_module.py: This code implements interface with the sentence selection model to score sentence pairs.
- Sentence_Selection_Port_Test.ipynb: This notebook tests the sentence selection model as an importable module.

