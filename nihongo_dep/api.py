from flask import Flask
from flask_cors import CORS

from . import ginza, tree

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})


@app.route("/parse/<string:input>")
def parse(input: str):
    ginza_data = ginza.GinzaDataFrame.from_text(input)
    return dict(
        text=input,
        tokens=ginza_data.reset_index()
        .drop(columns=["token"])
        .to_dict(orient="records"),
        phrases=tree.parse_tree(ginza_data),
        type="sentence",
    )
