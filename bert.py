import sys
import codecs
import numpy as np
from keras_bert import load_trained_model_from_checkpoint
import tokenization

class BertEngine():
    folder = 'multi_cased_L-12_H-768_A-12'
    config_path = folder+'/bert_config.json'
    checkpoint_path = folder+'/bert_model.ckpt'
    vocab_path = folder+'/vocab.txt'

    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_path, do_lower_case=False)
    model = load_trained_model_from_checkpoint(config_path, checkpoint_path, training=True)
    #model.summary()

    def fillup_missed(self, sentence):
        sentence = sentence.replace(' [MASK] ','[MASK]')
        sentence = sentence.replace('[MASK] ','[MASK]')
        sentence = sentence.replace(' [MASK]','[MASK]')
        sentence = sentence.split('[MASK]')

        tokens = ['[CLS]']

        for i in range(len(sentence)):
            if i == 0:
                tokens = tokens + self.tokenizer.tokenize(sentence[i]) 
            else:
                tokens = tokens + ['[MASK]'] + self.tokenizer.tokenize(sentence[i]) 
        tokens = tokens + ['[SEP]']
        

        token_input = self.tokenizer.convert_tokens_to_ids(tokens)
        token_input = token_input + [0] * (512 - len(token_input))

        mask_input = [0]*512
        for i in range(len(mask_input)):
            if token_input[i] == 103:
                mask_input[i] = 1

        seg_input = [0]*512

        token_input = np.asarray([token_input])
        mask_input = np.asarray([mask_input])
        seg_input = np.asarray([seg_input])

        predicts = self.model.predict([token_input, seg_input, mask_input])[0] 
        predicts = np.argmax(predicts, axis=-1)
        predicts = predicts[0][:len(tokens)]
        out = []

        for i in range(len(mask_input[0])):
            if mask_input[0][i] == 1:                       # [0][i], т.к. сеть возвращает batch с формой (1,512), где в первом элементе наш результат
                out.append(predicts[i]) 

        out = self.tokenizer.convert_ids_to_tokens(out)          # индексы в текстовые токены
        out = ' '.join(out)                                 # объединяем токены в строку с пробелами
        out = tokenization.printable_text(out)              # в удобочитаемый текст
        out = out.replace(' ##','') 
        
        return out

    def followup(self, sentence1, sentence2):
        tokens_sen_1 = self.tokenizer.tokenize(sentence1)
        tokens_sen_2 = self.tokenizer.tokenize(sentence2)

        tokens = ['[CLS]'] + tokens_sen_1 + ['[SEP]'] + tokens_sen_2 + ['[SEP]']
        
        token_input = self.tokenizer.convert_tokens_to_ids(tokens)      
        token_input = token_input + [0] * (512 - len(token_input))
        
        mask_input = [0] * 512

        seg_input = [0]*512
        len_1 = len(tokens_sen_1) + 2                   # длина первой фразы, +2 - включая начальный CLS и разделитель SEP
        for i in range(len(tokens_sen_2)+1):            # +1, т.к. включая последний SEP
            seg_input[len_1 + i] = 1                    # маскируем вторую фразу, включая последний SEP, единицами

        token_input = np.asarray([token_input])
        mask_input = np.asarray([mask_input])
        seg_input = np.asarray([seg_input])

        predicts = self.model.predict([token_input, seg_input, mask_input])[1]

        return int(round(predicts[0][0]*100))







