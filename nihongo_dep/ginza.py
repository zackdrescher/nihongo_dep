import pandas as pd
import spacy
from spacy.tokens import Doc


def apply(data: pd.Series):
    nlp = spacy.load("ja_ginza")
    return data.apply(lambda x: nlp(x))


class GinzaDataFrame(pd.DataFrame):
    def __init__(self, data: Doc, sent_id: str = None):
        super().__init__(dict(tokens=list(data)))

        self["i"] = self.tokens.apply(lambda token: token.i)
        self["orth"] = self.tokens.apply(lambda token: token.orth_)
        self["lemma"] = self.tokens.apply(lambda token: token.lemma_)
        self["norm"] = self.tokens.apply(lambda token: token.norm_)
        self["morph_reading"] = self.tokens.apply(
            lambda token: token.morph.get("Reading")
        )
        self["pos"] = self.tokens.apply(lambda token: token.pos_)
        self["morph_inflection"] = self.tokens.apply(
            lambda token: token.morph.get("Inflection")
        )
        self["tag"] = self.tokens.apply(lambda token: token.tag_)
        self["dep"] = self.tokens.apply(lambda token: token.dep_)
        self["head"] = self.tokens.apply(lambda token: token.head.i)

        if sent_id is not None:
            self["sent_id"] = sent_id
