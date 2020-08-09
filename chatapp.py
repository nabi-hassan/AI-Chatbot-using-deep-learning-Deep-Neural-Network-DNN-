
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle
import json
import random
import tflearn
import numpy as np
import tensorflow
import nltk
from nltk.stem.lancaster import LancasterStemmer
from flask import Flask,render_template,request
from flask_wtf import FlaskForm
from flask_restful import Api, Resource, reqparse
from wtforms import StringField,SubmitField

def bag_of_words(s,words):
    stemmer = LancasterStemmer()

    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for ss in s_words:
        for i, w in enumerate(words):
            if w == ss:
                bag[i] = 1
    return np.array(bag)

app = Flask(__name__)

app.config['SECRET_KEY'] = 'mysecretkey'

class Chatbot(FlaskForm):
    ExampleInputEmail = StringField("Friend Chatbot")
    submit = SubmitField('Send')

@app.route('/')
def index():
    form = Chatbot()
    return render_template('chat.html',u_inp="",b_resp="",form=form)

@app.route('/chat', methods=['GET','POST'])
def chat():
    form = Chatbot()
    stemmer = LancasterStemmer()

    with open("dataset.json",encoding="utf8") as file:
        data = json.load(file)

    words = []
    with open("data.pickle","rb") as f:
        words,labels,training,output = pickle.load(f)

    tensorflow.compat.v1.reset_default_graph()
    net = tflearn.input_data(shape=[None, len(training[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
    net = tflearn.regression(net)

    model = tflearn.DNN(net)

    model.load("model.tflearn")

    inp = form.ExampleInputEmail.data
    form.ExampleInputEmail.data = ''
    if inp!=None:
        """if inp.lower() == "quit" or "end":
            return None"""
        print("hello")
        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = np.argmax(results)
        tag = labels[results_index]
        
        for tg in data["intents"]:
            if tg["tag"] == tag:
                responses = tg["responses"]
        resp = random.choice(responses)   
    
        return render_template('chat.html',u_inp=inp,b_resp=resp,form=form)
    return render_template('chat.html',u_inp="",b_resp="",form=form)


if __name__ == "__main__":
    app.run(debug=True)