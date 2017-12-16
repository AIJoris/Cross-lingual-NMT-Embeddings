# Cross-lingual-NMT-Embeddings
This repo forms a refactor of the code for a project that focussed on Neural Machine Translation 
and uses a modified version of OpenNMT.

##Paper abstract:
High dimensional representations of sentences and words have proven extremely useful in NMT, but can not be used for cross-language learning when models are separately trained. This work investigates the possibility of cross-lingual word or sentence representations. An attempt has been made to encode sentences and words from different languages (English and Dutch) to identical semantic space using a NMT model and an autoencoder with equal and frozen decoder weights. Results indicate that even though the NMT model and autoencoder have good performance, sentence and word cosine similarity measures between the models show that no identical semantic spaces are learned.

##Model usage:

    python OpenNMT-py/translate.py -model models/autoencoder.pt -src data/test_nl -output pred_autoencoder.txt -replace_unk -verbose
    python OpenNMT-py/translate.py -model models/translator-en-nl.pt -src data/test_en -output pred_translator.txt -replace_unk -verbose
    
translate.py, Translator.py and train.py have been modified.
