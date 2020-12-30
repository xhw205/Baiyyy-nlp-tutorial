# -*- coding: utf-8 -*-
# @Author  : Xhw
# @Date    : 2020/12/29
# @Note: 对中文支持，特别是医疗文本这种专业领域， 支持很差
#

import json
import codecs
import plac
import random
import warnings
from pathlib import Path
import spacy
from spacy.language import Language
from spacy.util import minibatch, compounding
import argparse

def tranfer_pattern(txt_path):
    res = []
    train_data_txt = codecs.open(txt_path, encoding='utf-8')
    for idx, item in enumerate(train_data_txt):
        if item.startswith(u'\ufeff'):  # 去掉BOM编码
            item = item.encode('utf-8')[3:].decode('utf-8')
        item = json.loads(item)

        raw_txt = item['originalText']
        entities = item["entities"]
        dic_entity = {'entities': []}

        for entity in entities:
            label_type = entity['label_type']  # 实体类别 包括: 疾病和诊断/手术/解剖部位等
            start_pos = entity['start_pos']
            end_pos = entity['end_pos']
            _ = entity['overlap'] #ignore
            dic_entity['entities'].append((int(start_pos), int(end_pos), label_type))
        single_data = (raw_txt, dic_entity)
        res.append(single_data)

    return res


def train(model=None, output_dir=None, num_iter=100, train_data=None):
    """Load the model, set up the pipeline and train the entity recognizer."""
    nlp = spacy.load(model)
    assert isinstance(nlp, Language)
    ner = nlp.get_pipe("ner")
    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions] #禁止其他 pipes 参与NER的训练

    with nlp.disable_pipes(*other_pipes):
        if model == None:
            return

        for itn in range(num_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,  # batch of texts
                    annotations,  # batch of annotations
                    drop=0.5,  # dropout - make it harder to memorise data
                    losses=losses,
                )
            print("Losses", losses)

    # for text, _ in TRAIN_DATA:
    #     doc = nlp(text)
    #     print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
    #     print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])

    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./data/test.txt', type=str)
    parser.add_argument('--model_path', default="D:\\NLP_Model\\spacy\\zh_model", type=str)
    parser.add_argument('--num_iter', default=200, type=int, help="Spacy epoch nums")
    parser.add_argument('--output_dir', default='./outputs', type=str)

    args = parser.parse_args()
    print('args:\n' + args.__repr__())

    TRAIN_DATA = tranfer_pattern(args.data_path)
    train(args.model_path, output_dir=args.output_dir, num_iter=args.num_iter, train_data=TRAIN_DATA)
