#!/usr/bin/env python
# coding: utf-8
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Util Functions
from util_funcs import *
from prepare_data import process_text_str    ## PRE PROCESSING
from topic_mining import give_topics_to_text ## TOPIC MINING

select_list_pk = 'pickles/select_list_dic.pk'

def load_select_list_items():
    return read_pickle(select_list_pk)
    
def get_topics_for_text(str_txt):
    cleansed_text = process_text_str(str_txt)
    return give_topics_to_text (cleansed_text[1])

if __name__ == "__main__":
    start_logger()
    printTS (len(load_select_list_items()))
    printTS (get_topics_for_text('Hello I am Asad. I am a good guy. I love chinese food. But my kids are naughty. Therefore I do not go often to restaurants, and mostly eat ah home. But when I go to restaurant, I order turkish food like Kababs etc. Then I get a bug tummy. The Kebab place in downtown Bern is nice. But it is rather small. Service was better earlier, but after change of management, not anymore. Any there are also no toilets there. But food, is good.'))
    
    