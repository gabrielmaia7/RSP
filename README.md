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

This directory contains:

- *data*:
  - bert_dev_from_fresh_trained_with_batch_size_32.json: This file contains the dev partition from FEVER with claim-sentence pairs given a relevance score by the model, for evaluating the model's performance on FEVER. Dev is the dev partition with golden data added to it (pairs which are KNOWN to be relevant are given a score of 1, other pairs are given the model's output as score). Generated by the model.best.32.pt model.
  - bert_eval_from_fresh_trained_with_batch_size_32.json: Same as above, but no golden data is added.
  - bert_test_from_fresh_trained_with_batch_size_32.json: Same as both above, but for the testing partition and WITH golden data.
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

## textual_entailment

This directory holds the scripts, by-product data, and models that consist of ProVe's textual entailment module. This module was trained from scratch by creating a training/testing dataset from FEVER. This directory includes thus many versions of the model, going from iterations on dataset construction, training strategies, and utilisation of hyperparameter tuning. It also includes different models.

This directory contains:

- *data*: This directory holds the training/evaluation/validation data created from FEVER to develop the textual entailment model.
  - *support_data_v1*: This directory includes all the data created according to the first strategy of drawing completely random sentences to represent the "not enough information" class.
    - train_wikidecoded.csv: The training data taken directly from FEVER, but with sentence IDs decoded into actual sentences based on the corresponding Wikipedia dumps pointed out by FEVER as sources.
    - shared_task_test_wikidecoded.csv: Same as above, but testing partition.
    - shared_task_dev_wikidecoded.csv: Same as above, but validation/dev partition.
    - blind_submission_bert_test_from_fresh_trained_with_batch_size_32_LABELLED_MALON.jsonl: Blind submission to be evaluated [here](https://competitions.codalab.org/competitions/18814#participate), representing the TER model's performance on the FEVER dataset's test partition by using the rule-based MALON strategy as aggregation. "Fresh trained with batch size 32" refers to the fact that sentence selection was obtained with the freshly trained sentence selection model with batch size = 32 found in the sentence_retrieval directory.
    - blind_submission_bert_test_from_fresh_trained_with_batch_size_32_LABELLED_WEIGHTED_SUM.jsonl: Same as above, but using the weighted sum strategy as aggregation.
    - bert_dev_from_fresh_trained_with_batch_size_32_with_support_scores_v1.csv: This file is the dev partition from FEVER (dev has sentence selection golden data added to it) with TER scores assigned by the TER model to each claim-sentence pair. Outputed by the notebook '2_Wikidecoded_trained_model_evaluation' and scores generated by the v1 model.
    - bert_eval_from_fresh_trained_with_batch_size_32_with_support_scores_v1.csv: Same as above, dev partition but without golden data added to it.
    - bert_test_from_fresh_trained_with_batch_size_32_with_support_scores_v1.csv: Same as above, but for the test partition and WITH golden sentence selection data.
  - *support_data_v2*: This directory includes all the data created according to the second strategy of getting "not enough information" sentences from a closely related page to the original claim.
    - bert_dev_from_fresh_trained_with_batch_size_32_with_support_scores_v2.csv: Same as v1, but produced by the v4_PBT model by the notebook '5_SentenceRetrievalBERT_trained_model_evaluation'.
    - bert_eval_from_fresh_trained_with_batch_size_32_with_support_scores_v2.csv: Same as above, but eval.
    - bert_test_from_fresh_trained_with_batch_size_32_with_support_scores_v2.csv: Same as above, but test.
    - blind_submission_bert_test_from_fresh_trained_with_batch_size_32_LABELLED_MALON.jsonl: Same as v1, but produced by the v4_PBT model by the notebook '5_SentenceRetrievalBERT_trained_model_evaluation'.
    - blind_submission_bert_test_from_fresh_trained_with_batch_size_32_LABELLED_WEIGHTED_SUM.jsonl:  Same as above, but for the test partition.
    - dev_support_from_bert.csv: The v2 dev data with the new strategy.
    - dev_support_from_bert_SPECIAL_CHARS_CODED.csv: Same as above but some special characters are coded as a sequence of normal characters.
    - test_support_from_bert.csv: The v2 test data with the new strategy.
    - test_support_from_bert_SPECIAL_CHARS_CODED.csv: Same as above but some special characters are coded as a sequence of normal characters.
    - train_support_from_bert.csv: The v2 train data with the new strategy.
    - train_support_from_bert_SPECIAL_CHARS_CODED.csv: Same as above but some special characters are coded as a sequence of normal characters.
- *models*: This directory holds all the models tested for the TER task.
  - textual_entailment_BERT_FEVER_v1.tar.gz: Found [here](https://emckclac-my.sharepoint.com/:u:/g/personal/k20036346_kcl_ac_uk/EfLpO1nOgoNGssgnm7_X4iABhV-FgK1X9LpoaIoutheqlA?e=Xl2xA2), contains the v1 trained model and tokenizers created with the data from ../data/support_data_v1 in the notebook '1_Using_BERT_on_FEVER_with_Trainer_3_Classes.ipynb'.
  - textual_entailment_BERT_FEVER_v2.tar.gz: Found [here](https://emckclac-my.sharepoint.com/:u:/g/personal/k20036346_kcl_ac_uk/EfGRmNFgbklHkoop1ulNmVYBriWx989iWryOZ_dZPsVIOg?e=Wp8B4X), contains the v2 trained model and tokenizers created with the data from ../data/support_data_v2/, both the versions with and without special chars coded, created in the notebook '4_Using_BERT_on_FEVER_with_Trainer_3_Classes_DATASET_V2.ipynb'.
  - textual_entailment_BERT_FEVER_v3.tar.gz: Found [here](https://emckclac-my.sharepoint.com/:u:/g/personal/k20036346_kcl_ac_uk/EXm8pHTIOYhOgHJ2xhe2ZNwB1VY0QUg3PEWiE8P_NHYSjA?e=eouaRA), contains the v3 trained model and tokenizers created similarly to v2, but with different hyper parameters further described in the notebook '7_Using_BERT_on_FEVER_with_Trainer_3_Classes_DATASET_V2_HYPER'.
  - textual_entailment_BERT_FEVER_v4_PBT.tar.gz: Found [here](https://emckclac-my.sharepoint.com/:u:/g/personal/k20036346_kcl_ac_uk/ETaBYgCbRUNOvAct_5jWSL0BCSfx34jPEkX5mWgo9BnGBA?e=luVbhN), contains the v4 trained model and tokenizers created similarly to v2/v3, but through PBT hyper-parameter exploration as described in the notebook '7_Using_BERT_on_FEVER_with_Trainer_3_Classes_DATASET_V2_HYPER'.
  - textual_entailment_BERT_FEVER_v4_ASHA.tar.gz: Found [here](https://emckclac-my.sharepoint.com/:u:/g/personal/k20036346_kcl_ac_uk/EdRl3CGbxz9Nju31dKpwgr0B7SgDC6ngre3C1L8FjjGzcw?e=LwNm2O), contains the v4 trained model and tokenizers created similarly to v2/v3, but through ASHA hyper-parameter exploration as described in the notebook '7_Using_BERT_on_FEVER_with_Trainer_3_Classes_DATASET_V2_HYPER'.
- *run*: This directory holds data and images obtained from the training/execution of the TER models, such as metrics/results and temporary model checkpoints.
- 0.5_Wikipedia_fever_corpus_to_sql.ipynb: This notebook converts the wikipedia dumps made available by FEVER into an SQL databse for easier retrieval.
- 0_fever_decode_wikipage_IDs_to_sentences.ipynb: This notebook creates the v1 data found in ./data/support_data/v1.
- 1_Using_BERT_on_FEVER_with_Trainer_2_Classes.ipynb: This notebook uses the v1 data to create a binary class model for testing whether the data can train a simpler model.
- 1_Using_BERT_on_FEVER_with_Trainer_3_Classes.ipynb: Same as above, but creates a ternary classifier. This generates the v1 model and tokenizer.
- 2_Wikidecoded_trained_model_evaluation.ipynb: This notebook evaluates the v1 model/tokenizer.
- 3_fever_processing_for_textual_entailment.ipynb: This notebook improves the way we generate the training data, creating the v2 data.
- 4_Using_BERT_on_FEVER_with_Trainer_3_Classes_DATASET_V2.ipynb: This notebook trains a new classifier on the v2 data, creating the v2 model/tokenizer.
- 5_SentenceRetrievalBERT_trained_model_evaluation.ipynb: This notebook evaluates the v2 model/tokenizer.
- 6_SentenceRetrievalBERT_trained_model_evaluation-MEASURE_MODELS_LAST_YEAR.ipynb: This notebook verifies if the models trained before and after migrating the data from one computation cluster to another differ in performance.
- 7_Using_BERT_on_FEVER_with_Trainer_3_Classes_DATASET_V2_HYPER.ipynb: This notebook uses different hyper parameters to generate the v3 model/tokenizer. It also uses hyper-parameter tuning approaches (PBT and ASHA) to create the v4 models/tokenizers.
- textual_entailment_df.json: This is WTR's 409 triple-reference pairs after exiting ProVe's textual entailment module.
- textual_entailment_module.py: This script implements utilities to help access the model and its functionalities.

**Note**: For a list of the conda environment under which this repository was developed, check *environment.yml*.
