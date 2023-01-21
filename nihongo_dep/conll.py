import pandas as pd
from pyconll.unit.conll import Conll
from pyconll.unit.sentence import Sentence


class ConllDataFrame(pd.DataFrame):
    def __init__(self, data: Conll):
        super().__init__(dict(sent=list(data)))
        self["id"] = self.sent.apply(lambda x: x.id)
        self["text"] = self.sent.apply(lambda x: x.text)

    def get_tokens(self):
        return (
            pd.concat([SentenceDataFrame(x, x.id) for x in self.sent])
            .reset_index()
            .rename(columns={"index": "sent_idx"})
        )


class SentenceDataFrame(pd.DataFrame):
    @classmethod
    def from_conll(cls, data: Conll):
        return (
            pd.concat([cls(x, x.id) for x in data])
            .reset_index()
            .rename(columns={"index": "sent_idx"})
        )

    def __init__(self, data: Sentence, sent_id: str = None):
        super().__init__(dict(tokens=list(data)))
        self["text"] = self.tokens.apply(lambda x: x.form)
        self["head"] = self.tokens.apply(lambda x: x.head)
        self["upos"] = self.tokens.apply(lambda x: x.upos)
        self["deprel"] = self.tokens.apply(lambda x: x.deprel)
        self["deps"] = self.tokens.apply(lambda x: x.deps)
        self["feats"] = self.tokens.apply(lambda x: x.feats)
        self["misc"] = self.tokens.apply(lambda x: x.misc)

        if sent_id is not None:
            self["sent_id"] = sent_id
