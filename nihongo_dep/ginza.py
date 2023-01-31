import ginza
import pandas as pd
import spacy
from spacy.tokens import Doc

nlp = spacy.load("ja_ginza_electra")


def apply(text: str):
    return nlp(text)


def apply_series(data: pd.Series):
    return data.apply(lambda x: nlp(x))


class GinzaDataFrame(pd.DataFrame):
    @classmethod
    def from_text(cls, text: str):
        return cls(apply(text))

    @classmethod
    def from_texts(cls, texts: pd.Series):
        sents = pd.DataFrame(dict(text=texts, ginza=apply_series(texts)))
        return sents, cls.from_docs(sents.ginza)

    @classmethod
    def from_docs(cls, docs: pd.Series):
        return pd.concat([cls(x, i) for i, x in docs.items()]).reset_index()

    def __init__(self, data: Doc, sent_id: str = None):
        super().__init__(dict(token=list(data)))

        self.set_index(
            self.token.apply(lambda token: token.i).rename("sent_idx"),
            inplace=True,
        )

        self["text"] = self.token.apply(lambda token: token.text)
        self["orth"] = self.token.apply(lambda token: token.orth_)
        self["lemma"] = self.token.apply(lambda token: token.lemma_)
        self["norm"] = self.token.apply(lambda token: token.norm_)
        self["morph_reading"] = self.token.apply(
            lambda token: token.morph.get("Reading")
        )
        self["pos"] = self.token.apply(lambda token: token.pos_)
        self["morph_inflection"] = self.token.apply(
            lambda token: token.morph.get("Inflection")
        )
        self["tag"] = self.token.apply(lambda token: token.tag_)
        self["dep"] = self.token.apply(lambda token: token.dep_)
        self["head"] = self.token.apply(lambda token: token.head.i)

        self["bunsetu_label"] = ginza.bunsetu_bi_labels(data)
        self["bunsetu_position"] = ginza.bunsetu_position_types(data)

        if sent_id is not None:
            self["sent_id"] = sent_id
