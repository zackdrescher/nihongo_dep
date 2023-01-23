import pandas as pd
import spacy
from spacy.tokens import Doc


def apply(data: pd.Series):
    nlp = spacy.load("ja_ginza_electra")
    return data.apply(lambda x: nlp(x))


class GinzaDataFrame(pd.DataFrame):
    @classmethod
    def from_text(cls, text: str):
        return cls(apply(pd.Series([text])))

    @classmethod
    def from_texts(cls, texts: pd.Series):
        return cls.from_docs(apply(texts))

    @classmethod
    def from_docs(cls, docs: pd.Series):
        return pd.concat([cls(x, i) for i, x in docs.items()]).reset_index()

    def __init__(self, data: Doc, sent_id: str = None):
        super().__init__(dict(token=list(data)))

        self.set_index(
            self.token.apply(lambda token: token.i).rename("sent_idx"),
            inplace=True,
        )

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

        if sent_id is not None:
            self["sent_id"] = sent_id
