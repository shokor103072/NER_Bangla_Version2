# NER_Bangla_Version2


Bangla Name Entity Recognition (`NER`) using [SpaCy](https://spacy.io/). NER has been employed on Bengali sentences, with a specific focus on identifying entities tagged as `'Person'` or `'PER'`. Our experiment utilizes a  [Transformer](https://en.wikipedia.org/wiki/Transformer_(machine_learning_model))-based model, which achieves a commendable `F1 score ~ 0.80` for this dataset

Bangla NER data is collected from, 
- [Rifat1493/Bengali-NER](https://raw.githubusercontent.com/Rifat1493/Bengali-NER/master/all_data.txt)
- [banglakit/bengali-ner-data](https://raw.githubusercontent.com/banglakit/bengali-ner-data/master/main.jsonl)

### Dependency
```bash
conda install spacy=3.1
pip install spacy-transformers # need if you want to use transformer 
```

### Data prepration

1. To automate data prepration just run,
```bash
python utils/convert_to_spacy_json_format.py
```
This scrip will generate `data/train.json`, `data/val.json`

2. Convert json data to `.spacy` data
```sh
python -m spacy convert data/train.json ./data
python -m spacy convert data/val.json ./data

# Outputs
✔ Generated output file (8986 documents): data/train.spacy
✔ Generated output file (999 documents): data/val.spacy
```
Above two command will generate `data/train.spacy`, and `data/val.spacy`

### Training & inferance of SpaCy Transition base model

#### Training 
#### Create base config file

Go to the [link](https://spacy.io/usage/training#config) and create a base config file and save it uinder `./configs/base_config.cfg`

![](./docs/images/spacy_base_config_file.png)

Required fils for ner task are already in,

```sh
configs/
├── base_config.cfg     # base ner file configuration download from spacy website
├── config.cfg          # use to train ner pipeline
└── config_pretrain.cfg # use to train only tok2vec seperately

```

#### Prepare configuration files
Now convert `./configs/base_config.cfg` to config file `./configs/config.cfg`

```
python -m spacy init fill-config configs/base_config.cfg configs/config.cfg
```

#### Start training
```bash
python -m spacy train configs/config.cfg \
    --output ./models \
    --paths.train ./data/train.spacy \
    --paths.dev ./data/val.spacy
```
You will get F1 score on val data around `0.66`

#### Inferance
For inferance please run,
```bash
python test.py
```
You can already pretrain model in `test.py`. Please download the pretrain model from [google drive (4.4MB)](https://drive.google.com/file/d/1IqF87JGlClqPsU7I7Et5lvi_BZksnKDl/view?usp=sharing) and set the model path in `test.py` file


### Training and inferance SpaCy transformer pipeline
To training spacy transformer model please check `need GPU`,

[Transformer training and inferance guide](./transformers/readme.md)

You will get F1 score on val data around `0.80`


#### Transformer based model sample output

if you want to use already trained model please download pretrain model from [google drive (622.8MB)](https://drive.google.com/drive/folders/1tGxN2fmaOFx7FK__Udr7WB02snhUL7_v?usp=sharing) and set the model path in `test.py` file

```python

import spacy
from spacy.lang.bn import Bengali

# Create a blank Bengali model
nlp = Bengali()

# Add the transformer component
nlp.add_pipe("transformer")

# Load English tokenizer, tagger, parser and NER
nlp = spacy.load("/models_multilingual_bert/model-best")

text_list = [
    "আব্দুর রহিম নামের কাস্টমারকে একশ টাকা বাকি দিলাম",
    "১০০ টাকা জমা দিয়েছেন কবির",
    "ডিপিডিসির স্পেশাল টাস্কফোর্সের প্রধান মুনীর চৌধুরী জানান",
    "অগ্রণী ব্যাংকের জ্যেষ্ঠ কর্মকর্তা পদে নিয়োগ পরীক্ষার প্রশ্নপত্র ফাঁসের অভিযোগ উঠেছে।",
    "সে আজকে ঢাকা যাবে",
]
for text in text_list:
    doc = nlp(text)

    print(f"Input: {text}")
    for entity in doc.ents:
        print(f"Entity: {entity.text}, Label: {entity.label_}")
    print("---")
# Outputs
    Input: আব্দুর রহিম নামের কাস্টমারকে একশ টাকা বাকি দিলাম
    Entity: আব্দুর রহিম, Label: PER
    ---
    Input: ১০০ টাকা জমা দিয়েছেন কবির
    Entity: কবির, Label: PER
    ---
    Input: ডিপিডিসির স্পেশাল টাস্কফোর্সের প্রধান মুনীর চৌধুরী জানান
    Entity: মুনীর চৌধুরী, Label: PER
    ---
    Input: অগ্রণী ব্যাংকের জ্যেষ্ঠ কর্মকর্তা পদে নিয়োগ পরীক্ষার প্রশ্নপত্র ফাঁসের অভিযোগ উঠেছে।
    ---
    Input: সে আজকে ঢাকা যাবে
    ---
```

__NOTE__: [Why to use Transformer base model ?](https://spacy.io/usage/embeddings-transformers#vectors-vs-language-models)

### NER Data formats
```
BLIOU data format meaning
B = Begin
L = Last
I = Inside
O = Outside
U = Unique
```

```
IOB data format meaning
I = Inside
O = Outside
B = Begin
```

### References
- [BILOU data formats meaning](https://stackoverflow.com/questions/17116446what-do-the-bilou-tags-mean-in-named-entity-recognition)
- [SpaCy 3.1 data format](https://zachlim98.github.io/me/2021-03/spacy3-ner-tutorial)
- [Preparing the training data](https://spacy.io/usage/training#training-data)
- [Performance throught uncertainty](https://saxamos.github.io/2020/07/31/en-improve-spacy-performance-through-uncertainty/)
- [WikiANN model](https://huggingface.co/datasets/wikiann)
- [Transformer machine learning model](https://en.wikipedia.org/wiki/Transformer_(machine_learning_model))
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

