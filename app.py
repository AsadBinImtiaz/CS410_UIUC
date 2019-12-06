#!/usr/bin/env python
# coding: utf-8
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, flash, redirect,url_for, render_template, request, session, abort, make_response, render_template_string
from src.util_funcs import *
from src.webapp import *
from django.utils.encoding import smart_str

from flask import Flask, render_template_string

app = Flask(__name__)

template = "analyse.html"
states = load_select_list_items()
state=""
city=""
rest=""
revw=""

@app.route('/')
def index():

    return render_template(template, states=states)

@app.route('/analyse',methods = ['POST', 'GET'])
def analyse():
    
    result=""
    state =""
    city=""
    rest=""
    revw=""
    if request.method == 'POST':
        print('inPost')
        rlst = request.form['inpts'].split(';')
        text=rlst[0]
        rest=rlst[1]
        city=rlst[2]
        state=rlst[3]
        revw=text
        if len(text) == 0:
            result = "Please select a review ID"
            print(text)
   
        else:
            result = text
            
    return render_template("analyse.html",states=states,result=result,state=state,city=city,rest=rest,revw=revw)

@app.route('/play',methods = ['POST', 'GET'])
def play():
    result=""
    review=""
    if request.method == 'POST':
        text = request.form['inpts']
        review = text
        if len(text) == 0:
            result = "Please select a review"
            print(text)
   
        else:
            result = text
            
    return render_template("play.html",result=result, review=review)

app.run()