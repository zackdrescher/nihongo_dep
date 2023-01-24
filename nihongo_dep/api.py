from flask import Flask

from . import ginza

app = Flask(__name__)


@app.route("/parse/<string:input>")
def parse(input: str):
    return dict(text=input, tokens=ginza.GinzaDataFrame.from_text(input))
