# Cross-lingual-NMT-Embeddings
This repo forms a refactor of the code for a project that focussed on Neural Machine Translation 
and uses a modified version of OpenNMT. 

Model usage:

    python OpenNMT-py/translate.py -model models/autoencoder.pt -src data/test_nl -output pred_autoencoder.txt -replace_unk -verbose
    python OpenNMT-py/translate.py -model models/translator-en-nl.pt -src data/test_en -output pred_translator.txt -replace_unk -verbose
