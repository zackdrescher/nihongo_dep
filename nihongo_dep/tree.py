import pandas as pd


def get_root(df: pd.DataFrame):
    root = df[df.dep == "ROOT"]
    assert len(root) == 1
    root = root.iloc[0]
    return root


def get_children(tokens: pd.DataFrame, node: pd.Series):
    children = tokens[tokens["head"] == node.name].drop(node.name, errors="ignore")
    return children


def get_leading_tokens(ix: int, tokens):
    leading = []

    while ix - 1 in tokens.index:
        ix -= 1
        leading.insert(0, tokens.loc[ix])

    return leading


def get_trailing_tokens(ix: int, tokens):
    trailing = []

    while ix + 1 in tokens.index:
        ix += 1
        trailing.append(tokens.loc[ix])

    return trailing


def get_bunsetsu_phrase(root, tokens):
    ix = root.name

    leading = get_leading_tokens(ix, tokens)
    trailing = get_trailing_tokens(ix, tokens)

    return pd.DataFrame(leading + [root] + trailing)


def parse_node(node: pd.Series, tokens: pd.DataFrame):
    children = get_children(tokens, node)

    phrase = get_bunsetsu_phrase(node, children)

    remaining = children.drop(phrase.index, errors="ignore")

    out = dict(
        phrase=phrase.text.str.cat(),
        tokens=phrase.drop(columns=["token"]).to_dict("records"),
    )

    if not remaining.empty:
        out["children"] = [parse_node(x, tokens) for _, x in remaining.iterrows()]

    return out


def parse_tree(tokens: pd.DataFrame):
    root = get_root(tokens)

    return parse_node(root, tokens)
