import sys
import codecs
import numpy as np
from keras_bert import load_trained_model_from_checkpoint
import tokenization
import bert


bert_engine = bert.BertEngine()

#sentence = 'Я пришел в [MASK] и купил [MASK].'
#print(sentence)
#res = bert_engine.fillup_missed(sentence)
#print('Result:', res)

res = bert_engine.followup("Я пришел в магазин.", "И купил молоко.")
print(res)








