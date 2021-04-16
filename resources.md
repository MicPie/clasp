## Resources


### Models

* For protein models: 
    * ESM-1b is the current state of the art for protein embeddings, MSA-transformer if we have MSA data (i think we wont): https://github.com/facebookresearch/esm (requires torch.hub or direct download of weights (several Gigabytes))

* For biomed-related models (via HuggingFace): 
    * The Roberta model for ChemProt (w/ 84.4 score) task: https://huggingface.co/allenai/dsp_roberta_base_dapt_biomed_tapt_chemprot_4169/tree/main (paper from april 2020: https://arxiv.org/pdf/2004.10964v1.pdf)
    * The roberta for biomed language (same paper as above) (w/ 83.0 in ChemProt but i guess more general?): https://huggingface.co/allenai/biomed_roberta_base
    * The PubMedBERT: https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext (ChemProt score of 77.24 but tested on more datasets as well) (arxiv from august 2020 - https://arxiv.org/pdf/2007.15779.pdf)

### Datasets

* For zero-shot prediction: 
   * Mutation effect: https://raw.githubusercontent.com/FowlerLab/Envision2017/master/data/clinvar_predicted_2017-03-21.csv
   * Protein functional families (derived from FunFams DB) (each row contains: family number - uniprot_ids) (approx 185 max prots per class) https://github.com/Rostlab/FunFamsConsensus/blob/master/data/funfam160_uniprot_mapping.txt) 
