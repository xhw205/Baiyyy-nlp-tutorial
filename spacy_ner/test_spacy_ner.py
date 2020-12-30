# -*- coding: utf-8 -*-
# @Author  : Xhw
# @Date    : 2020/12/30
# @FileName: test_spacy_ner.py
import spacy
from spacy_ner import tranfer_pattern
nlp2 = spacy.load('./outputs')

for text, _ in tranfer_pattern('./data/test.txt'):
    doc = nlp2(text)
    print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
    print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])
