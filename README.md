# Baiyyy-nlp-tutorial

医疗文本 NLP 任务示例

+ python 3.7+
+ pytorch 1.0+

## Medical NER

### Spacy NER

数据：存放在 spacy_ner/data，[Yidu-S4K：医渡云结构化4K数据集](https://github.com/lrs1353281004/Chinese_medical_NLP#1-yidu-s4k%E5%8C%BB%E6%B8%A1%E4%BA%91%E7%BB%93%E6%9E%84%E5%8C%964k%E6%95%B0%E6%8D%AE%E9%9B%86)

模型：下载 spacy 中文支持模型 [地址](https://github.com/explosion/spacy-models/releases/download/zh_core_web_md-2.3.1/zh_core_web_md-2.3.1.tar.gz) ，解压后如下面结构

```
/spacy/zh_model    
  | - meta.json                 # 模型描述信息           
   ....
  | - parser                    # 依存分析模型
  | - ner                       # 命名实体识别模型
```

#### Training

$``cd spacy_ner``

$``python spacy_ner.py --model_path xxx_yours_path``

#### Testing

$``python test_spacy_ner.py``

#### **Note**

+ 训练会报 “实体未对齐”的 warnings

  > 这是由于 spacy 对医疗文本分词支持很差，且该警告无法忽略

+ 测试表明 spacy 并不适用于 医疗文本这种专业领域

  