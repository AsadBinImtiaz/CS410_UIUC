#!/usr/bin/env python
# coding: utf-8
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Util Functions
from util_funcs import *
from prepare_data import process_text_str                     ## PRE PROCESSING
from topic_mining import give_topics_to_text                  ## TOPIC MINING
from sentiment_analysis import get_sentiment_score, text_process            ## Sentiment analysis
from lara import give_aspects_to_text, give_selected_aspects  ## LARA analysis

select_list_pk = 'pickles/select_list_dic.pk'
aspect_list_pk = 'pickles/asp_aspects_app.pk'
busnes_list_pk = 'pickles/asp_review_app.pk'
review_list_pk = 'pickles/asp_text_app.pk'

def load_select_list_items():
    return read_pickle(select_list_pk)
    
def load_review_list_items():
    aspect_df   = read_pickle(aspect_list_pk)
    business_df = read_pickle(busnes_list_pk)
    review_df   = read_pickle(review_list_pk)
    
    merged_df = business_df
    merged_df['topic_text'] = review_df['topic_text']
    merged_df['sentiment_text'] = review_df['sentiment_text']
    
    Aspects = ['atmosphere','food','location','service','staff','value']
    
    for asp in Aspects:
        merged_df[asp] = aspect_df[asp]
        
    return merged_df
    
   
def get_topics_for_text(str_txt):
    cleansed_text = process_text_str(str_txt)
    return give_topics_to_text (cleansed_text[1])

def get_topics_for_cleaned_text(str_txt):
    return give_topics_to_text (str_txt)

def get_senti_score_for_text(str_txt):
    return get_sentiment_score (str_txt)

def get_clean_senti_score_for_text(str_txt):
    return get_sentiment_score (process_text_str(str_txt))
    
def get_aspects_for_text(str_txt):
    return give_aspects_to_text  (str_txt)

def get_selected_aspects(str_txt):
    return give_selected_aspects(str_txt)

def get_clean_aspects_score_for_text(str_txt):
    cleansed_text = process_text_str(str_txt)
    return give_clean_aspects_to_text (cleansed_text[1])

#def get_weighted_aspects_scores_(lst_scores):
#    return lst_scores

def get_score_color(res):
    rep = ""
    res=str(res)
    if res=='1':
        rep = '<b><font color="#7B241C">'+str(res)+'</font></b>'
    if res=='2':
        rep = '<b><font color="#E67E22">'+str(res)+'</font></b>'
    if res=='3':
        rep = '<b><font color="#0000FF">'+str(res)+'</font></b>'
    if res=='4':
        rep = '<b><font color="#00FF00">'+str(res)+'</font></b>'
    if res=='5':
        rep = '<b><font color="#008000">'+str(res)+'</font></b>'
    return str(rep)

def get_result_body(res,topic,ptopic,ntopic):
    str_txt = get_result_body_template();
    str_txt = str_txt.replace('strRestaurantName',res['name'])
    str_txt = str_txt.replace('strRestaurantReview',str(res['text']).replace('\\n','<br/>').replace('\n','<br/>'))
    str_txt = str_txt.replace('strRestaurantScore',str(res['review_stars']))
    str_txt = str_txt.replace('strCalculatedScore',get_score_color(res['sentScore']))
    str_txt = str_txt.replace('strAtmsAspect',str(res['atmosphere']).replace('\n','<br/>'))
    str_txt = str_txt.replace('strFoodAspect',str(res['food']).replace('\n','<br/>'))
    str_txt = str_txt.replace('strServAspect',str(res['service']).replace('\n','<br/>'))
    str_txt = str_txt.replace('strStffAspect',str(res['staff']).replace('\n','<br/>'))
    str_txt = str_txt.replace('strLocnAspect',str(res['location']).replace('\n','<br/>'))
    str_txt = str_txt.replace('strValuAspect',str(res['value']).replace('\n','<br/>'))
    str_txt = str_txt.replace('strAtmsScore',get_score_color(res['atmosphereScore']))
    str_txt = str_txt.replace('strFoodScore',get_score_color(res['foodScore']))
    str_txt = str_txt.replace('strServScore',get_score_color(res['serviceScore']))
    str_txt = str_txt.replace('strStffScore',get_score_color(res['staffScore']))
    str_txt = str_txt.replace('strLocnScore',get_score_color(res['locationScore']))
    str_txt = str_txt.replace('strValuScore',get_score_color(res['valueScore']))
    str_txt = str_txt.replace('strTopicTxt','<font color="#0000FF">'+str(topic)+'</font>')
    str_txt = str_txt.replace('strTopicPos','<font color="#008000">'+str(ptopic)+'</font>')
    str_txt = str_txt.replace('strTopicNeg','<font color="#FF0000">'+str(ntopic)+'</font>')
    return str_txt

def get_result_body_analyse(str_rev,reviews_df):
    printTS(f"Called: get_result_body_analyse called: {str_rev} {len(reviews_df)}")
    str_txt = ""
    if str_rev != "":
        row = reviews_df[(reviews_df['review_id'] == str_rev)]
        if len(row) == 1:
            res = give_selected_aspects(row)
            topics_List = give_topics_to_text (row['sentiment_text'])
            if int(res['sentScore']) >3:
                topics_List[2] = ''
            if int(res['sentScore']) <3:
                topics_List[1] = ''
            str_txt=get_result_body(res,topics_List[0],topics_List[1],topics_List[2])
    return str_txt
        
def get_result_body_play(str_txt=""):
    printTS(f"Called: get_result_body_play called: {len(str_txt)}")
    str_rev = ""
    if str_txt != "":
        cleansed_text = process_text_str(str_txt)
        topics_List = give_topics_to_text (cleansed_text[1])
        
        printTS(f"Topic Miner Returned: {topics_List}")
        
        res = give_aspects_to_text(str_txt)
        
        #printTS(f"Aspect Miner Returned: {res} for {str_txt}")
        if len(res) > 0:
            if int(res['sentScore']) >3:
                topics_List[2] = ''
            if int(res['sentScore']) <3:
                topics_List[1] = ''
            str_txt=get_result_body(res,topics_List[0],topics_List[1],topics_List[2])    

    return str_txt
    
#def get_result_body_filled():
def get_result_body_template():
    return """
                            <div><table id="restable" style="width: 100%">
								<tr id="titlerow">
									<div><p style="font-size: 20px; color: #CB4335;"><b>Analysis Result:</b></p></div>
								</tr>
								<tr id="inforow">
								    <div>
									<table style="width: 100%">
										<tr>
											<td style="width: 15%">
												<label style="text-align: left"><b>Restaurant Name:</b></label>
											</td>
											<td style="width 85%">
												<label style="text-align: left"><table><tr>strRestaurantName</tr></table></label>
											</td>
											</td>
										</tr>
										<tr>
											<td style="width: 15%; vertical-align: top;">
												<label style="text-align: left; vertical-align: top;"><b>Review Text:</b></label>
											</td>
											<td style="width 85%; margin-right: 15%;">
												<label style="text-align: left; margin-right: 15%;"><table><tr>strRestaurantReview</tr></table></label>
											</td>
											</td>
										</tr>
										<tr>
											<td style="width: 15%">
												<label style="text-align: left"></label>
											</td>
											<td>
												<table style="width: 30%">
													<tr>
													<td style="width: 15%; text-align: left">Review Score:</td>
													<td style="width: 10%; text-align: left"><b>strRestaurantScore</b></td>
													</tr>
													<tr>&nbsp;&nbsp;<br/></tr>
													<tr>
													<td style="width: 15%; text-align: left">Calculated Score:</td>
													<td style="width: 10%; text-align: left"><b>strCalculatedScore</b></td>
													</tr>
												</table>
											</td>
										</tr>
									</table>
								    </div>
								    <div>
									<table style="width: 100%">
										<tr>&nbsp;&nbsp</tr>
										<tr style="background-color: #cde8ef;">
											<td style="width: 10%"><div style="text-align: left"><b>Aspects</b><br/></div></td>
											<td style="width: 15%"><div style="text-align: center"><label style="text-align: center"><b>Food</b></label></div></td>
											<td style="width: 15%"><div style="text-align: center"><label style="text-align: center"><b>Atmosphere</b></label></div></td>
											<td style="width: 15%"><div style="text-align: center"><label style="text-align: center"><b>Service</b></label></div></td>
											<td style="width: 15%"><div style="text-align: center"><label style="text-align: center"><b>Staff</b></label></div></td>
											<td style="width: 15%"><div style="text-align: center"><label style="text-align: center"><b>Location</b></label></div></td>
											<td style="width: 15%"><div style="text-align: center"><label style="text-align: center"><b>Value</b></label></div></td>
										</tr>
										<tr>&nbsp;&nbsp;</tr>
										<tr>
											<td style="width: 10%; background-color: #cde8ef;"><div style="text-align: left"><label style="text-align: center"><b>Text Segments</b></label></div></td>
											<td style="width: 15%; text-align: left; vertical-align: top"><div>strFoodAspect</div></td>
											<td style="width: 15%; text-align: left; vertical-align: top"><div>strAtmsAspect</div></td>
											<td style="width: 15%; text-align: left; vertical-align: top"><div>strServAspect</div></td>
											<td style="width: 15%; text-align: left; vertical-align: top"><div>strStffAspect</div></td>
											<td style="width: 15%; text-align: left; vertical-align: top"><div>strLocnAspect</div></td>
											<td style="width: 15%; text-align: left; vertical-align: top"><div>strValuAspect</div></td>
										</tr>
										<tr>&nbsp;&nbsp;</tr>
										<tr>
											<td style="width: 10%; background-color: #cde8ef;"><div style="text-align: left"><label style="text-align: center"><b>Aspect Score<b></label></div></td>
											<td style="width: 15%"><div style="text-align: center;">strFoodScore</div></td>
											<td style="width: 15%"><div style="text-align: center;">strAtmsScore</div></td>
											<td style="width: 15%"><div style="text-align: center;">strServScore</div></td>
											<td style="width: 15%"><div style="text-align: center;">strStffScore</div></td>
											<td style="width: 15%"><div style="text-align: center;">strLocnScore</div></td>
											<td style="width: 15%"><div style="text-align: center;">strValuScore</div></td>
										</tr>
									</table>
									</div>
									<div>
									<table style="width: 100%">
										<tr>&nbsp;&nbsp</tr>
										<tr>
											<td style="width: 15%">
												<label style="text-align: left"><b>Main Topic:</b></label>
											</td>
											<td style="width 85%">
												<label style="text-align: left"><table><tr>strTopicTxt</tr></table></label>
											</td>
											</td>
										</tr>
										<tr>
											<td style="width: 15%">
												<label style="text-align: left"><b>Positive Topic:</b></label>
											</td>
											<td style="width 85%">
												<label style="text-align: left"><table><tr>strTopicPos</tr></table></label>
											</td>
											</td>
										</tr>
										<tr>
											<td style="width: 15%">
												<label style="text-align: left"><b>Nagative Topic:</b></label>
											</td>
											<td style="width 85%">
												<label style="text-align: left"><table><tr>strTopicNeg</tr></table></label>
											</td>
											</td>
										</tr>
									</table>
									</div>
								</tr>
							</table></div>"""

if __name__ == "__main__":
    start_logger()
    printTS (len(load_select_list_items()))
    printTS (get_topics_for_text('Hello I am Asad. I am a good guy. I love chinese food. But my kids are naughty. Therefore I do not go often to restaurants, and mostly eat ah home. But when I go to restaurant, I order turkish food like Kababs etc. Then I get a bug tummy. The Kebab place in downtown Bern is nice. But it is rather small. Service was better earlier, but after change of management, not anymore. Any there are also no toilets there. But food, is good.'))
    
    