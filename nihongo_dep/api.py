from flask import Flask
from flask_cors import CORS

from . import ginza

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})


@app.route("/parse/<string:input>")
def parse(input: str):
    return dict(
        text=input,
        tokens=ginza.GinzaDataFrame.from_text(input)
        .drop(columns=["token"])
        .to_dict(orient="records"),
    )
