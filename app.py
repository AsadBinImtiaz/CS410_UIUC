#!/usr/bin/env python
# coding: utf-8
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, flash, redirect,url_for, render_template, request, session, abort, make_response
from src.util_funcs import *
from django.utils.encoding import smart_str

app = Flask(__name__)

@app.route('/')
def index():
	return render_template("analyse.html",**locals())

if __name__ == "__main__":
    printTS("Started")
    
    app.run(debug = False)