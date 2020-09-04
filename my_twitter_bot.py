import tweepy
import time
import torch
from transformers import AutoTokenizer, AutoModelWithLMHead

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
model = AutoModelWithLMHead.from_pretrained("microsoft/DialoGPT-large")

from keys import *

auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_KEY, ACCESS_SECRET)
api = tweepy.API(auth)

FILE_NAME = 'last_seen_id.txt'

def retrieve_last_seen_id(file_name):
    f_read = open(file_name, 'r')
    last_seen_id = int(f_read.read().strip())
    f_read.close()
    return last_seen_id

def store_last_seen_id(last_seen_id, file_name):
    f_write = open(file_name, 'w')
    f_write.write(str(last_seen_id))
    f_write.close()
    return

def reply_to_tweets():
    #tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
    #model = AutoModelWithLMHead.from_pretrained("microsoft/DialoGPT-large)
    print('retrieving and replying to tweets...', flush=True)
    # DEV Note: use 1294080888771239937 for testing
    last_seen_id = retrieve_last_seen_id(FILE_NAME)
    # note we need to use tweet_mode='extended' below to show 
    # all full tweets (with full_text). without it, long tweets
    # would be cut off.
    mentions = api.mentions_timeline(last_seen_id, tweet_mode='extended')
    for mention in reversed(mentions):
        print(str(mention.id) + ' - ' + mention.full_text, flush=True)
        last_seen_id = mention.id
        store_last_seen_id(last_seen_id, FILE_NAME)
        length = 280 + len(mention.full_text)
        encoded_output = tokenizer.encode(mention.full_text + tokenizer.eos_token, return_tensors='pt')
        chat_history_ids = model.generate(encoded_output, max_length=length, temperature=1.0, top_k=0, top_p=0.9,
            repetition_penalty=1.0, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(chat_history_ids[:, encoded_output.shape[-1]:][0], skip_special_tokens=True)
        api.update_status('@' + mention.user.screen_name + " " + response, mention.id)
count = 0
while count <= 40:
    reply_to_tweets()
    print("count: " + str(count))
    time.sleep(15)
    count+=1
