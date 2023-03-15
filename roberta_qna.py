
# import flask
# from flask import request, jsonify
# from flask_cors import CORS, cross_origin
import logging
from transformers import RobertaTokenizer, RobertaForQuestionAnswering
import json
import numpy as np
from elasticsearch import Elasticsearch
from pprint import pprint
from copy import deepcopy
from transformers import BertTokenizer, RobertaTokenizer
from base64 import b64encode
import torch 
import streamlit as st

# Define the Streamlit app
st.title("Document based Question Answer")
question_txt = st.text_input("Enter your question:")

roberta_tokenizer = RobertaTokenizer.from_pretrained("deepset/roberta-large-squad2")
roberta_model = RobertaForQuestionAnswering.from_pretrained("deepset/roberta-large-squad2")

es = Elasticsearch(["http://localhost:9200"], \
                   headers={"Authorization": "Basic " + b64encode("elastic:kf3X2=O2gM3opG9kN3tq".encode()).decode()})



# question = "what is DHCP server and how to configure DHCP server?"
question = str(question_txt)
model = "router"
make = "cisco"
index="test_index_v4_roberta"

def query_ES(question, model, make, index, verbose=False):
    """
    Runs search query on ElasticSearch; retuns response
    """
    body ={
          "query": {
                  "match": {
                    "text": {
                      "query": "what is DHCP server and how to configure DHCP server?"
                    }
                  }
          },
        "size": 5
        }
    res = es.search(index=index, body=body)
    if verbose:
        print(res['hits']['total'])
        print('Top result (pages):', [x['_source']["page"] for x in res['hits']['hits']])
    return res


def run_QA_model(context, question, model, tokenizer, verbose=False):
    """
    Runs given QA model with given question and context; returns answer string and tokens
    context - context string for QA model
    question - question string for QA model
    model - QA model
    tokenizer - tokenizer for QA model
    """
    encoding = tokenizer.encode_plus(question, context)
    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]
    if verbose:
        print("Number of input tokens: ", len(input_ids))
        
    if len(input_ids) > 512:
        print(question)
        print(context)
        print("too many input tokens.. skipped")
        raise RuntimeError
    outputs = model(torch.tensor([input_ids]), attention_mask=torch.tensor([attention_mask]))
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    ans_tokens = input_ids[torch.argmax(start_scores) : torch.argmax(end_scores)+1]
    answer_tokens = tokenizer.convert_ids_to_tokens(ans_tokens , skip_special_tokens=True)
    answer_tokens_to_string = tokenizer.convert_tokens_to_string(answer_tokens)
    return answer_tokens_to_string, answer_tokens_to_string.replace("\n", " ")

res = query_ES(question, model, make, 'test_index_v4_roberta')

for doc in res['hits']['hits']:
    context = doc['_source']['text']
    try:
        ans, ans_tokens = run_QA_model(context, question, roberta_model, roberta_tokenizer)
    except RuntimeError as err:
        print(doc['_source']['page'])
        print('------------------------------------\n')
        continue
    
    if(len(ans_tokens) > 1):
        # print("page: ", doc['_source']['page'])
        st.write("Answer:", ans)
        st.write("--------------------------\n")

        # print ("Question ",question)
        # print ("\nAnswer : ", ans)
        # print('------------------------------------\n')
                
    

    
