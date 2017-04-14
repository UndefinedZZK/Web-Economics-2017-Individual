from __future__ import division
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import string
import math
import csv
import random
import pandas as pd
import numpy as np

#load data from file, and prepare to deliver it to train the model
def load_data(origin_data):
    data = defaultdict(list)
    labels = defaultdict(list)
    #nomalize the slotprice using the standard deviation
    STD_SLOTPRICE = origin_data.slotprice.values.std()
    
    for click, weekday, hour, useragent, region, city, adexchange, slotwidth, slotheight, slotvisibility, payprice, slotformat, slotprice, advertiser, usertag in origin_data[['click', 'weekday', 'hour', 'useragent', 'region', 'city', 'adexchange', 'slotwidth', 'slotheight', 'slotvisibility', 'payprice', 'slotformat', 'slotprice', 'advertiser', 'usertag']].values:
        instance = encoding_data(STD_SLOTPRICE, weekday, hour, useragent, region, city, adexchange, slotwidth, slotheight, slotvisibility, slotformat, slotprice, advertiser, usertag, payprice)
        data[str(advertiser)].append(instance)
        labels[str(advertiser)].append(int(click))

    return data,labels

#Use Logistic Regression to train the model
def train_CTR_model(training_events, training_labels):
    model = {}
    #key is different advertisers
    for key in training_events.keys():
        data = training_events[key]
        labels = training_labels[key]

        label_encoder = LabelEncoder()
        vectorizer = DictVectorizer()

        training_event_x = vectorizer.fit_transform(data)
        training_event_y = label_encoder.fit_transform(labels)

        #caculate the weight to balance data
        neg_weight = sum(labels) / len(labels)
        pos_weight = 1 - neg_weight

        #fit data and train model
        lr = LogisticRegression(class_weight = {1: pos_weight, 0: neg_weight})
        lr.fit(training_event_x, training_event_y)
        model[key] = (lr, label_encoder, vectorizer)
    return model

#encode the data, pick only the features we are going to use to train the model
def encoding_data(STD_SLOTPRICE, weekday, hour, useragent, region, city, adexchange, slotwidth, slotheight, slotvisibility, slotformat, slotprice, advertiser, usertag, payprice = '', ):
    instance = {'weekday': str(weekday), 'hour': str(hour), 'useragent': str(useragent), 'region': str(region), \
                'city': str(city), 'adexchange': str(adexchange), 'slotwidth': str(slotwidth), 'slotheight': str(slotheight), \
                'slotvisibility': str(slotvisibility), 'payprice':payprice, 'slotformat': str(slotformat), 'slotprice': float(slotprice) / STD_SLOTPRICE, \
                'advertiser': str(advertiser)}
    #user tag splits to different tags, so need to add each of them
    usertags = usertag.split(',')
    usertags_dict = {}
    for each_usertag in usertags:
        usertags_dict["usertag " + each_usertag] = True
    instance.update(usertags_dict)
    return instance

#use the model to predict CTR
def getting_pCTR(instance, advertiser, model): 
    lr = model[advertiser][0]
    label_encoder = model[advertiser][1]
    vectorizer = model[advertiser][2]
    event = [instance]
    event_x = vectorizer.transform(event)
    event_y = lr.predict_proba(event_x)
    return event_y[0][1]


def generating_bid_value(model, origin_data, avgCTR, base_bid, numerator = 1000, start_budget = 25000000):  
    impressions = 0
    clicks = 0
    budget=start_budget
    STD_SLOTPRICE = origin_data.slotprice.values.std()
    results=[]

    for click, bidid, weekday, hour, useragent, region, city, adexchange, slotwidth, slotheight, slotvisibility, payprice, slotformat, slotprice, advertiser, usertag in origin_data[['click', 'bidid', 'weekday', 'hour', 'useragent', 'region', 'city', 'adexchange', 'slotwidth', 'slotheight', 'slotvisibility', 'payprice', 'slotformat', 'slotprice', 'advertiser', 'usertag']].values:
        instance = encoding_data(STD_SLOTPRICE, weekday, hour, useragent, region, city, adexchange, slotwidth, slotheight, slotvisibility, slotformat, slotprice, advertiser, usertag, payprice)
        pCTR = getting_pCTR(instance,str(advertiser), model)
        current_bid = base_bid * ((pCTR/avgCTR)**(numerator/1000))
        if budget > current_bid and current_bid > payprice:
            impressions += 1
            budget -= payprice
            clicks += click

    print "Base bid :" + str(base_bid)
    print "Impressions :" + str(impressions)
    print "Clicks: " + str(clicks)
    print "Cost: " + str((25000000 - budget))
    if impressions > 0:
        print "CTR: " + str((clicks / impressions))
        print "CPC: " + str((25000000 - budget)/impressions)
        return impressions, clicks, 25000000 - budget
    else:
        return -1,-1,-1



def generating_bid_value_for_test(model, origin_data, avgCTR, base_bid, numerator = 1000): 
    STD_SLOTPRICE = origin_data.slotprice.values.std()

    bid_id = []
    bid_price = []

    for bidid, weekday, hour, useragent, region, city, adexchange, slotwidth, slotheight, slotvisibility, slotformat, slotprice, advertiser, usertag in origin_data[['bidid', 'weekday', 'hour', 'useragent', 'region', 'city', 'adexchange', 'slotwidth', 'slotheight', 'slotvisibility', 'slotformat', 'slotprice', 'advertiser', 'usertag']].values:
        instance = encoding_data(STD_SLOTPRICE, weekday, hour, useragent, region, city, adexchange, slotwidth, slotheight, slotvisibility, slotformat, slotprice, advertiser, usertag)
        pCTR = getting_pCTR(instance,str(advertiser), model)
        current_bid = base_bid * ((pCTR/avgCTR)**(numerator/1000))
        bid_id.append(bidid)
        bid_price.append(current_bid)
    return bid_id, bid_price
    

training_path = r"../dataset/train.csv"
validation_path = r"../dataset/validation.csv"
test_path = r"../dataset/test.csv"


training_data = pd.read_csv(training_path)
validation_data = pd.read_csv(validation_path)
test_data = pd.read_csv(test_path)

avgCTR = (training_data.click.sum() / training_data.shape[0]) * 100

training_events, training_labels = load_data(training_data)
model = train_CTR_model(training_events, training_labels)

Bid = pd.DataFrame()
#set the base bid to 13
Bid['Base_Bid'] = np.arange(13,14)

im = []
clks = []
ct = []
for base_bid in Bid['Base_Bid']:
#base_bid = 13
#for numerator in range(980,991):
    numerator = 990
    #print numerator
    [imps, clicks, cost] = generating_bid_value(model, validation_data, avgCTR, base_bid, numerator, 25000000)
    im.append(imps)
    clks.append(clicks)
    ct.append(cost) 

    [bid_id, bid_price] = generating_bid_value_for_test(model, test_data, avgCTR, base_bid, numerator)
Bid['Impressions'] = im
Bid.Impressions = Bid.Impressions.astype(int)
Bid['Spend'] = ct
Bid['Clicks'] = clks
Bid['CTR'] = (Bid.Clicks/Bid.Impressions ).round(4).astype(str)
Bid['CPM'] = (Bid.Spend/Bid.Impressions).round(2).astype(str)
Bid['CPC'] = (Bid.Spend/Bid.Clicks).round(2).astype(str)
print Bid
Bid.to_csv('Evaluation Bid.csv')

Bid_Result = pd.DataFrame()
Bid_Result['bidid'] = bid_id
Bid_Result['bidprice'] = bid_price
print Bid_Result
Bid_Result.to_csv(r"output.csv")