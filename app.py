#!/usr/bin/env python


import json
import os
import logging

from common import file
from flask import Flask
from flask import request
from flask import make_response
from textpredict.text_classification_predict import TextClassificationPredict

app = Flask(__name__)

#@app.route('/webhook', methods=['POST'])
@app.route('/')

def hello():
    """Return a friendly HTTP greeting."""
    return 'Hello World!'

@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500

def webhook():
    req = request.get_json(silent=True, force=True)

    db_train = file.get_dbtrain()
    db_train_extend = file.get_dbtrain_extend()
    db_answers = file.get_dbanswers()

    res = makeWebhookResult(req, db_train, db_train_extend, db_answers)
    res = json.dumps(res, indent=4)

    r = make_response(res)
    r.headers['Content-Type'] = 'application/json'
    return r


def makeWebhookResult(req, db_train, db_train_extend, db_answers):
    if req.get("result").get("action") != "chatbot-vietnamairline":
        return {}

    result = req.get("result")
    txt = result.get("resolvedQuery")

    tcp = TextClassificationPredict(txt, db_train, db_train_extend, db_answers,)
    speech = tcp.Text_Predict()

    return {
        "speech": speech,
        "displayText": speech,
        "source": req.get("result").get("action")
    }


if __name__ == '__main__':
    #port = int(os.getenv('PORT', 8080))
    #print("Starting app on port %d" %(port))
    #app.run(debug=True, port=port, host='0.0.0.0')

    app.run(host='127.0.0.1', port=8080, debug=True)