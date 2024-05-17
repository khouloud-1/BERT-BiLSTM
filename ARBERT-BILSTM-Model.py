# -*- coding: utf-8 -*-

"""
Created on Tue Feb 15 17:27:03 2022

@author: BOUCHIHA

"""

import tensorflow as tf

import pandas as pd

df = pd.read_csv("WiHArD.csv", encoding='utf-8')
df['tpc']=df['category_code'].apply(lambda x: 0 if x==1 else (1 if x==2 else (2 if x==3 else (3 if x==4 else (4 if x==5 else (5 if x==6 else (6 if x==7 else (7 if x==8 else (8 if x==9 else (9 if x==10 else (10 if x==11 else 11)))))))))))

# Split it into training and test data set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['tpc'], test_size=0.3, train_size=0.7, stratify=df['tpc'])

# Arabic BERT Model
from transformers import BertTokenizer, BertModel

# ARBERT
BRT = "ARBERT"
tokenizer = BertTokenizer.from_pretrained("UBC-NLP/ARBERT")
Bmodel = BertModel.from_pretrained("UBC-NLP/ARBERT")

# Bert Model
def Arabic_Bert_Model_T(t_input):
    try:
        inputs = tokenizer(t_input, truncation=True, max_length = 512, return_tensors="pt")
        outputs = Bmodel(**inputs)
        print("Good input")
        return outputs.pooler_output.detach().numpy()[0]
    except:
        inputs = tokenizer("", return_tensors="pt")
        outputs = Bmodel(**inputs)
        print("Bad input")
        return outputs.pooler_output.detach().numpy()[0]

def Arabic_Bert_Model_G(ts_input):
    A = list()
    for i in ts_input:
            A.append(Arabic_Bert_Model_T(i))
    return tf.convert_to_tensor(A, dtype=tf.float32)


X_tr = Arabic_Bert_Model_G(X_train)
X_ts = Arabic_Bert_Model_G(X_test)

#set(y_train)

# BILSTM Model
from tensorflow.keras.utils import to_categorical
y_tr = to_categorical(y_train)
y_ts = to_categorical(y_test)

#from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dense, GlobalMaxPooling1D
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dense
#from tensorflow.keras.models import Model
#from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout
#from tensorflow.keras.layers import LSTM, Embedding, TimeDistributed, Dropout, Lambda

BILSTMmodel = Sequential()
# model.add(Embedding(input_dim=X_tr.shape[0], output_dim=X_tr.shape[1], input_length=X_tr.shape[1], weights=[X_tr], trainable=False))
input_shape = (768, 1)
BILSTMmodel.add(Input(shape=input_shape))
# model.add(Dropout(0.2))
BILSTMmodel.add(Bidirectional(LSTM(300, return_sequences=True)))
BILSTMmodel.add(Dropout(0.2))

BILSTMmodel.add(Bidirectional(LSTM(300, return_sequences=False)))
BILSTMmodel.add(Dropout(0.2))
BILSTMmodel.add(Dense(12, activation='softmax'))
BILSTMmodel.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')

#from keras.callbacks import EarlyStopping, ModelCheckpoint

#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)
#mc = ModelCheckpoint('./model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)


# Train the model (Training phase)
BILSTMmodel.fit(
    X_tr, y_tr,
    #validation_data=(X_ts, y_ts),
    epochs=50#,
    #callbacks=[tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=2)]
)


# Evaluate the model (Test phase)
from sklearn.metrics import classification_report
import numpy as np

y_predicted = BILSTMmodel.predict(X_ts)
y_predicted = np.argmax(y_predicted, axis=-1)


print("\n\n----------------  Flat Evaluation:  "+ BRT + "  -----------------------------------------------")
print(classification_report(y_test, y_predicted, zero_division=0))


# Hierarchical F score computation starts from here
import xml.etree.ElementTree as et
hierarchy = et.parse('WiHArD.hierarchy.xml').getroot()

def get_ancestors(root, val):  # Recursive
    # this function returns all parent classes IDs in a set format
    for element in root.findall('label'):
        if element is not None:
            if int(element.get('code')) == int(val):
                return set([int(element.get('code'))])
            r = get_ancestors(element, val)  # Recursion
            if r is not None:
                r.add(int(element.get('code')))
                return r

def get_hf_scores(y_true, y_pred):
    # y_true & y_pred are numpy arrays contain numbers between 0 and 11
    # this function to get the hierarchical f-score of the classifier
    t = 0
    tp = 0
    tr = 0
    
    for i in range(len(y_true)):
        t += len(get_ancestors(hierarchy, y_true[i]+1).intersection(get_ancestors(hierarchy, y_pred[i]+1)))  # aka |C' n C|
        tp += len(get_ancestors(hierarchy, y_pred[i]+1))  # aka |C'|
        tr += len(get_ancestors(hierarchy, y_true[i]+1))  # aka |C|
    h_p = t / tp  # hierarchical precision
    h_r = t / tr  # hierarchical recall
    h_f = (2 * h_p * h_r) / (h_p + h_r)  # hierarchical f-score
    return h_f, h_p, h_r

def print_hF_score(y_true, y_pred):
    h_f, h_p, h_r = get_hf_scores(y_true, y_pred)

    print('\nHierarchical measures:\n \tHPrecision = ' + str(h_p) + '\n \tHRecall = ' + str(
        h_r) + '\n \tHF-score = ' + str(h_f))

# Hierarchical F score computation ends here.

#added evaluation phase
import numpy as np

y_true = y_test.to_numpy()
classes = ['1- ثقافة','2- ملابس','3- طعام و شراب','4- سياحة','5- تاريخ','6- أحداث','7- إختراعات','8- آثار','9- رياضيات','10- جبر','11- تحليل','12- هندسة']






print_hF_score(y_true, y_predicted)
#end of evaluation


# Inference (Prediction phase)
print("\n\n\----------------  Prediction  -----------------------------------------------")
x = 'هذا التراث الثقافي غير المادي المتوارث جيلا عن جيل، تبدعه الجماعات والمجموعات من جديد بصورة مستمرة، بما يتفق مع بيئتها وتفاعلاتها مع الطبيعة وتاريخها، وهو ينمي لديها الإحساس بهويتها والشعور باستمراريتها، ويعزز من ثم احترام التنوع الثقافي والقدرة الإبداعية البشرية.'#input('Enter your the text to be classified: ')
msg = [x]
X_pr = Arabic_Bert_Model_G(msg)
r = BILSTMmodel.predict(X_pr)

# display the main category of the new text
main_topic_of_x = np.argmax(r[0])
print("\nMain class of the new text : ")
if (main_topic_of_x == 0): print('1- ثقافة')
elif  (main_topic_of_x == 1): print('2- ملابس')
elif  (main_topic_of_x == 2): print('3- طعام و شراب')
elif  (main_topic_of_x == 3): print('4- سياحة')
elif  (main_topic_of_x == 4): print('5- تاريخ')
elif  (main_topic_of_x == 5): print('6- أحداث')
elif  (main_topic_of_x == 6): print('7- إختراعات')
elif  (main_topic_of_x == 7): print('8- آثار')
elif  (main_topic_of_x == 8): print('9- رياضيات')
elif  (main_topic_of_x == 9): print('10- جبر')
elif  (main_topic_of_x == 10): print('11- تحليل')
elif  (main_topic_of_x == 11): print('12- هندسة')

# display all categories of the new text
r = r.flatten()
r = np.where(r > 0.8, 1, 0)
topics_of_x = np.flatnonzero(r == np.max(r))

print("\nAll classes of the new text : ")
for i in range(len(topics_of_x)):
    if (topics_of_x[i] == 0): print('1- ثقافة')
    elif  (topics_of_x[i] == 1): print('2- ملابس')
    elif  (topics_of_x[i] == 2): print('3- طعام و شراب')
    elif  (topics_of_x[i] == 3): print('4- سياحة')
    elif  (topics_of_x[i] == 4): print('5- تاريخ')
    elif  (topics_of_x[i] == 5): print('6- أحداث')
    elif  (topics_of_x[i] == 6): print('7- إختراعات')
    elif  (topics_of_x[i] == 7): print('8- آثار')
    elif  (topics_of_x[i] == 8): print('9- رياضيات')
    elif  (topics_of_x[i] == 9): print('10- جبر')
    elif  (topics_of_x[i] == 10): print('11- تحليل')
    elif  (topics_of_x[i] == 11): print('12- هندسة')



