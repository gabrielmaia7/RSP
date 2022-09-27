# ProVe's Pipeline Execution

ProVe is a pipelined tool for automated fact checking (AFC) on knowledge graphs (KGs), and is described in the paper "ProVe: A Pipeline for Automated Provenance Verification of Knowledge Graphs against Textual Sources".

This notebook organises the data, code, and models that make up ProVe. It also executes its different modules on WTR, an evaluation dataset consisting of KG triples and references extracted from Wikidata.

This repository has as main funcionatilites the following:
- Gathering and documenting the different modules/models that make up ProVe;
- Detailing the execution of ProVe on data extracted from Wikidata.

It consists of the following *directories*/files:
- support_classification_pipeline_test.ipynb: This ipython notebook contains the code which runs the ProVe pipeline on WTR, a set of KG triples and references sampled from Wikidata.
- *data*: This folder contains scripts that parse Wikidata, as well as data and scripts that compile the KG triples and references sampled from Wikidata, originally found disjoined in the 'text_extraction' directory, into an unannotated version of WTR (annotation is carried out in another Repo, alongside evaluation).
  - Wikidata_Claims_Parser.ipynb: A helper notebook which parses the triple and reference data from Wikidata
  - wikidata_parser.py: A script with helper functions to parse Wikidata
  - WTR_non_filtered_non_annotated.zip: A version of WTR obtained by joining the triple and reference data into a single dataset, but not yet annotated by the crowd
  - Compiling_WTR_Non_Annotated.ipynb: A notebook which takes the triple and reference CSV files available in the directory 'text_extraction' and joins them into a single non-annotated dataset.
- *text_extraction*: This directory holds the KG triple (claims) and reference data sampled from Wikidata which constitute the evaluation dataset WTR. These references have already been resolved to URLs and had their text parsed and extracted with sliding windows of size 1 and 2 (as described in the ProVe paper). This process was carried out at a [different repository](https://github.com/gabrielmaia7/WD_Textual_References_Dataset). More info can be found in this directory's README.
  - reference_html_as_sentences_df.csv: This file holds the 416 references samples from Wikidata, which represent WTR. For each reference, it shows their HTML code, the text retrieved from them, and its segmentation into multiple text segments which are then concatenated with sliding windows of size 1 and 2.
  - text_reference_claims_df.csv: This file holds all the claims (KG triples) extracted for each of the 416 references in 'reference_html_as_sentences_df.csv'.
- *verbalisation*: This directory holds the scripts, by-product data, and models that consist of ProVe's verbalisation module.
  - T5-Verbalisation-Test.ipynb: This notebook tests the verbalisation module found in the *graph2text* directory, ensuring it works properly.
  - verbalisation_module.py: This script encapsulates handy implementation to call the verbalisation model in *graph2text*.
  - verbalised_claims_df.csv: This dataset has, for each KG triple extracted in relation to one of the 416 references sampled from Wikidata, its component data, identifiers, and verbalisation as outputted by the model.
  - verbalisation_claims_df_main.csv: This is 'verbalised_claims_df.csv' after main verbalisations are chosen, which is to say one verbalisation per unique reference-triple pair with the preferred labels for each triple component.
  - verbalised_claims_df_main_corrected.csv: This is 'verbalised_claims_df_main.csv' after manual correction of those verbalisations that could not be improved by selecting alternative labels.
  - verbalised_claims_df_final.csv: This is 'verbalised_claims_df_main_corrected.csv' after ontological triples and duplicated verbalisation-url pairs were removed. Then, an unique triple for each reference is selected, resulting in the 409 triples-reference pairs that make up WTR, since 7 are dropped due to having duplicated verbalisation-url combinations.
  - *graph2text*: This repository holds the trained data verbalisation model. It is a slightly modified version of the [graph2text](https://github.com/UKPLab/plms-graph2text/) scripts by the folks at UKPLab, more specifically Ribeiro et al. We trained the model on webnlg as instructed in their repository, carrying here only the trained model and the code needed to run it. We have updated small bits of code as needed in order to adapt to more recent versions of pytorch and pytorch lightning. Some files were too big and were put on a cloud:
    - verbalisation/graph2text/outputs/t5-base_13881/val_avg_bleu=68.1000-step_count=5.ckpt: This file is not found in this repo and is instead found [here](https://emckclac-my.sharepoint.com/:u:/g/personal/k20036346_kcl_ac_uk/EbL1yTauXtpEqs4Izc97WNIBhumczrDGTNQb47uYGzXqsg?e=I9B5pR).
    - verbalisation/graph2text/outputs/t5-base_13881/best_tfmr/pytorch_model.bin: This file is not found in this repo and is instead found [here](https://emckclac-my.sharepoint.com/:u:/g/personal/k20036346_kcl_ac_uk/ES1YcFbwwIVJqpz0OcSnUsUBLvRnlV64AxtMiU8cczn9pw?e=yR56F3).
- 