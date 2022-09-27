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
