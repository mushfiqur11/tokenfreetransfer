# Text Representations for Cross-Lingual Transfer

This is the official repository of the paper "To token or not to token: A Comparative Study of Text Representations for Cross-Lingual Transfer"

Link to original [paper](https://aclanthology.org/2023.mrl-1.6/).

### Abstract: 

Choosing an appropriate tokenization scheme
is often a bottleneck in low-resource crosslingual transfer.
To understand the downstream implications of text representation choices,
we perform a comparative analysis on language models having diverse text representation modalities including 2 segmentationbased models (BERT, mBERT), 1 image-based
model (PIXEL), and 1 character-level model (CANINE). First, we propose a scoring Language Quotient (LQ) metric capable of providing a weighted representation of both zero-shot
and few-shot evaluation combined. Utilizing this metric, we perform experiments comprising 19 source 
languages and 133 target languages on three tasks (POS tagging, Dependency parsing, and NER). Our analysis reveals
that image-based models excel in cross-lingual transfer when languages are closely related and
share visually similar scripts. However, for tasks biased toward word meaning (POS, NER),
segmentation-based models prove to be superior. Furthermore, in dependency parsing tasks
where word relationships play a crucial role, models with their character-level focus, outperform others. Finally, we propose a recommendation scheme based on our findings to guide
model selection according to task and language
requirements


Please cite the following:

    @inproceedings{rahman-etal-2023-token,

        title = "To token or not to token: A Comparative Study of Text Representations for Cross-Lingual Transfer",
        
        author = "Rahman, Md Mushfiqur  and
          Sakib, Fardin Ahsan  and
          Faisal, Fahim  and
          Anastasopoulos, Antonios",
          
        editor = "Ataman, Duygu",
        
        booktitle = "Proceedings of the 3rd Workshop on Multi-lingual Representation Learning (MRL)",
        
        month = dec,
        
        year = "2023",
        
        address = "Singapore",
        
        publisher = "Association for Computational Linguistics",
        
        url = "https://aclanthology.org/2023.mrl-1.6",
        
        doi = "10.18653/v1/2023.mrl-1.6",
        
        pages = "67--84",
    
    }
