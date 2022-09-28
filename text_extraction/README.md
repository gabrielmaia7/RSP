The contents of this folder have been moved to another repository, [WD_Textual_References_Dataset](https://anonymous.4open.science/r/WD_Textual_References_Dataset-510B/), in which this task was taken as a bigger task of obtaining a suitable HTML-to-Text Wikidata Dataset for fact verification.

Firstly, no boilerplate removal was performed. We are trusting on the combination of the sentence selection module with a clean retrieval of well-formatted and significant natural language sentences to filter in important pieces of evidence, without the need to detect low-information and poor-formatted sentences in the HTML itself.

The final data files created in [WD_Textual_References_Dataset]([https://anonymous.4open.science/status/WD_Textual_References_Dataset-510B](https://anonymous.4open.science/r/WD_Textual_References_Dataset-510B/)) were copied to this folder, and will be what the pipeline will be tested on. They are:
- reference_html_as_sentences_df.csv: This file holds the 416 references samples from Wikidata, which represent WTR. For each reference, it shows their HTML code, the text retrieved from them, and its segmentation into multiple text segments which are then concatenated with sliding windows of size 1 and 2.
- text_reference_claims_df.csv: This file holds all the claims (KG triples) extracted for each of the 416 references in 'reference_html_as_sentences_df.csv'.
