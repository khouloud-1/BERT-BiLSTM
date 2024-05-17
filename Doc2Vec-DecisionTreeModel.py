# -*- coding: utf-8 -*-

"""

Created on Mon May 24 15:24:30 2021



@author: Djelloul BOUCHIHA, Abdelghani BOUZIANE, and Noureddine DOUMI

@paper: Machine learning for Arabic Text classification: Comparative study

"""



# -*- coding: utf-8 -*-

"""

Created on Fri May 21 19:30:00 2021



@author: Djelloul BOUCHIHA, Abdelghani BOUZIANE, Noureddine DOUMI and Mustafa JARRAR.

@paper: Machine learning for Arabic Text classification: Comparative study

"""



########################### Preprocessing ###############################



from nltk.corpus import stopwords



from textblob import TextBlob

import re



#from dsaraby import DSAraby

#ds = DSAraby()



from tashaphyne.stemming import ArabicLightStemmer



#from nltk.stem.isri import ISRIStemmer



stops = set(stopwords.words("arabic"))

stop_word_comp = {"،","آض","آمينَ","آه","آهاً","آي","أ","أب","أجل","أجمع","أخ","أخذ","أصبح","أضحى","أقبل","أقل","أكثر","ألا","أم","أما","أمامك","أمامكَ","أمسى","أمّا","أن","أنا","أنت","أنتم","أنتما","أنتن","أنتِ","أنشأ","أنّى","أو","أوشك","أولئك","أولئكم","أولاء","أولالك","أوّهْ","أي","أيا","أين","أينما","أيّ","أَنَّ","أََيُّ","أُفٍّ","إذ","إذا","إذاً","إذما","إذن","إلى","إليكم","إليكما","إليكنّ","إليكَ","إلَيْكَ","إلّا","إمّا","إن","إنّما","إي","إياك","إياكم","إياكما","إياكن","إيانا","إياه","إياها","إياهم","إياهما","إياهن","إياي","إيهٍ","إِنَّ","ا","ابتدأ","اثر","اجل","احد","اخرى","اخلولق","اذا","اربعة","ارتدّ","استحال","اطار","اعادة","اعلنت","اف","اكثر","اكد","الألاء","الألى","الا","الاخيرة","الان","الاول","الاولى","التى","التي","الثاني","الثانية","الذاتي","الذى","الذي","الذين","السابق","الف","اللائي","اللاتي","اللتان","اللتيا","اللتين","اللذان","اللذين","اللواتي","الماضي","المقبل","الوقت","الى","اليوم","اما","امام","امس","ان","انبرى","انقلب","انه","انها","او","اول","اي","ايار","ايام","ايضا","ب","بات","باسم","بان","بخٍ","برس","بسبب","بسّ","بشكل","بضع","بطآن","بعد","بعض","بك","بكم","بكما","بكن","بل","بلى","بما","بماذا","بمن","بن","بنا","به","بها","بي","بيد","بين","بَسْ","بَلْهَ","بِئْسَ","تانِ","تانِك","تبدّل","تجاه","تحوّل","تلقاء","تلك","تلكم","تلكما","تم","تينك","تَيْنِ","تِه","تِي","ثلاثة","ثم","ثمّ","ثمّة","ثُمَّ","جعل","جلل","جميع","جير","حار","حاشا","حاليا","حاي","حتى","حرى","حسب","حم","حوالى","حول","حيث","حيثما","حين","حيَّ","حَبَّذَا","حَتَّى","حَذارِ","خلا","خلال","دون","دونك","ذا","ذات","ذاك","ذانك","ذانِ","ذلك","ذلكم","ذلكما","ذلكن","ذو","ذوا","ذواتا","ذواتي","ذيت","ذينك","ذَيْنِ","ذِه","ذِي","راح","رجع","رويدك","ريث","رُبَّ","زيارة","سبحان","سرعان","سنة","سنوات","سوف","سوى","سَاءَ","سَاءَمَا","شبه","شخصا","شرع","شَتَّانَ","صار","صباح","صفر","صهٍ","صهْ","ضد","ضمن","طاق","طالما","طفق","طَق","ظلّ","عاد","عام","عاما","عامة","عدا","عدة","عدد","عدم","عسى","عشر","عشرة","علق","على","عليك","عليه","عليها","علًّ","عن","عند","عندما","عوض","عين","عَدَسْ","عَمَّا","غدا","غير","ـ","ف","فان","فلان","فو","فى","في","فيم","فيما","فيه","فيها","قال","قام","قبل","قد","قطّ","قلما","قوة","كأنّما","كأين","كأيّ","كأيّن","كاد","كان","كانت","كذا","كذلك","كرب","كل","كلا","كلاهما","كلتا","كلم","كليكما","كليهما","كلّما","كلَّا","كم","كما","كي","كيت","كيف","كيفما","كَأَنَّ","كِخ","لئن","لا","لات","لاسيما","لدن","لدى","لعمر","لقاء","لك","لكم","لكما","لكن","لكنَّما","لكي","لكيلا","للامم","لم","لما","لمّا","لن","لنا","له","لها","لو","لوكالة","لولا","لوما","لي","لَسْتَ","لَسْتُ","لَسْتُم","لَسْتُمَا","لَسْتُنَّ","لَسْتِ","لَسْنَ","لَعَلَّ","لَكِنَّ","لَيْتَ","لَيْسَ","لَيْسَا","لَيْسَتَا","لَيْسَتْ","لَيْسُوا","لَِسْنَا","ما","ماانفك","مابرح","مادام","ماذا","مازال","مافتئ","مايو","متى","مثل","مذ","مساء","مع","معاذ","مقابل","مكانكم","مكانكما","مكانكنّ","مكانَك","مليار","مليون","مما","ممن","من","منذ","منها","مه","مهما","مَنْ","مِن","نحن","نحو","نعم","نفس","نفسه","نهاية","نَخْ","نِعِمّا","نِعْمَ","ها","هاؤم","هاكَ","هاهنا","هبّ","هذا","هذه","هكذا","هل","هلمَّ","هلّا","هم","هما","هن","هنا","هناك","هنالك","هو","هي","هيا","هيت","هيّا","هَؤلاء","هَاتانِ","هَاتَيْنِ","هَاتِه","هَاتِي","هَجْ","هَذا","هَذانِ","هَذَيْنِ","هَذِه","هَذِي","هَيْهَاتَ","و","و6","وا","واحد","واضاف","واضافت","واكد","وان","واهاً","واوضح","وراءَك","وفي","وقال","وقالت","وقد","وقف","وكان","وكانت","ولا","ولم","ومن","مَن","وهو","وهي","ويكأنّ","وَيْ","وُشْكَانََ","يكون","يمكن","يوم","ّأيّان"}



name = "arabic-stop-words.txt"

aswf = open(name, 'r', encoding='utf-8')

counts = dict()

for line in aswf:

    words = line.split()

    for word in words:

        if word not in stop_word_comp:

            stop_word_comp.add(word)



#print(len(stop_word_comp))



ArListem = ArabicLightStemmer()



def stem(text_P):

    zen = TextBlob(text_P)

    words = zen.words

    cleaned = list()

    for w in words:

        cleaned.append(ArListem.light_stem(w))

    return " ".join(cleaned)



import pyarabic.araby as araby

def normalizeArabic(text_P):

    text_P = text_P.strip()

    text_P = re.sub("[إأٱآا]", "ا", text_P)

    text_P = re.sub("ى", "ي", text_P)

    text_P = re.sub("ؤ", "ء", text_P)

    text_P = re.sub("ئ", "ء", text_P)

    text_P = re.sub("ة", "ه", text_P)

    noise = re.compile(""" ّ    | # Tashdid

                             َ    | # Fatha

                             ً    | # Tanwin Fath

                             ُ    | # Damma

                             ٌ    | # Tanwin Damm

                             ِ    | # Kasra

                             ٍ    | # Tanwin Kasr

                             ْ    | # Sukun

                             ـ     # Tatwil/Kashida

                         """, re.VERBOSE)

    text_P = re.sub(noise, '', text_P)

    text_P = re.sub(r'(.)\1+', r"\1\1", text_P) # Remove longation

    return araby.strip_tashkeel(text_P)

    

def remove_stop_words(text_P):

    zen = TextBlob(text_P)

    words = zen.words

    return " ".join([w for w in words if not w in stops and not w in stop_word_comp and len(w) >= 2])





def clean_text(text_P):

    ## Remove punctuations

    text_P = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text_P)  # remove punctuation

    ## Remove stop words

    text_P = remove_stop_words(text_P)

    ## Remove numbers

    text_P = re.sub("\d+", " ", text_P)

    ## Remove Tashkeel

    text_P = normalizeArabic(text_P)

    #text_P = re.sub('\W+', ' ', text_P)

    text_P = re.sub('[A-Za-z]+',' ',text_P)

    text_P = re.sub(r'\\u[A-Za-z0-9\\]+',' ',text_P)

    ## remove extra whitespace

    text_P = re.sub('\s+', ' ', text_P)  

    #Stemming

    text_P = stem(text_P)

    return text_P





##################################  Read the corpus    ###########################################



import csv



import time 

start = time.time()



texts = list()

Y = list()



with open('WiHArD.csv', newline='', encoding='utf-8') as csvfile:

    reader = csv.DictReader(csvfile)

    for row in reader:

        line = clean_text(row['text']).split()

        Y.append(float(row['category_code']))

        texts.append(line)





from collections import defaultdict

frequency = defaultdict(int)

for text in texts:

    for token in text:

        frequency[token] += 1



# A term has to appear at least 1 time(s) in the corpus to be considered

processed_corpus = [[token for token in text if frequency[token] > 0] for text in texts]





############################### Doc2Vec #####################################



import gensim 

import numpy as np



# Create the tagged document needed for Doc2Vec

def create_tagged_document(list_of_list_of_words):

    for i, list_of_words in enumerate(list_of_list_of_words):

        yield gensim.models.doc2vec.TaggedDocument(list_of_words, [i])



train_data = list(create_tagged_document(processed_corpus))



#print(train_data[:])





# Init the Doc2Vec model /  vector_size corresponds to the number of features

model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)



# Build the Volabulary

model.build_vocab(train_data)



# Train the Doc2Vec model

model.train(train_data, total_examples=model.corpus_count, epochs=model.epochs)



D = list()

for i in processed_corpus:

    D.append(model.infer_vector(i))





y=np.array(D)

y = np.transpose(y[:, :])



D2V = np.vstack([y, Y])

D2V = np.transpose(D2V[:, :])





from sklearn.model_selection import train_test_split



XYtrain, XYtest = train_test_split(D2V, test_size=0.3, train_size=0.7, shuffle=True)





Xtrain = XYtrain[:,:XYtrain.shape[1]-1]

Ytrain = XYtrain[:,XYtrain.shape[1]-1:]

#print("\n****** Xtrain  ******")

#print(Xtrain)

#print("\n****** Ytrain  ******")

#print(Ytrain)





Xtest = XYtest[:,:XYtest.shape[1]-1]

Ytest = XYtest[:,XYtest.shape[1]-1:]

#print("\n****** Xtest  ******")

#print(Xtest)

#print("\n****** Ytest  ******")

#print(Ytest)





############################### DecisionTreeClassifier #####################################



from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()

clf = clf.fit(Xtrain, np.ravel(Ytrain))



#######################################



end = time.time()



############################### DecisionTreeClassifier evaluation #####################################



from sklearn.metrics import f1_score

y_true = Ytest[:,0]

y_pred = clf.predict(Xtest)



fs = f1_score(y_true, y_pred , average='micro')





print('\nCorpus size (dataset size): '+ str(D2V.shape[0]) + ' documents')

print('\nNumber of features (vector size): '+ str(D2V.shape[1]-1))

print('\nTime for preprocessing, the training data for Doc2Vec (features extraction) and DecisionTreeClassifier training: ',(end-start),' sec')

print('\nFor Doc2Vec and DecisionTreeClassifier, f1-score =                 '+str(fs))



############################### Classification Report #####################################



from sklearn.metrics import classification_report

print("\n-------------  Flat Evaluation:  ---------------------------- :")

print(classification_report(y_true, y_pred, zero_division=0))





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





def get_hf_scores():

    # this function to get the hierarchical f-score of the classifier



    t = 0

    tp = 0

    tr = 0

    # np_y_test = y_true.to_numpy

    for i in range(len(y_true)):

        t += len(

            get_ancestors(hierarchy, y_true[i]).intersection(get_ancestors(hierarchy, y_pred[i])))  # aka |C' n C|

        tp += len(get_ancestors(hierarchy, y_pred[i]))  # aka |C'|

        tr += len(get_ancestors(hierarchy, y_true[i]))  # aka |C|

    h_p = t / tp  # hierarchical precision

    h_r = t / tr  # hierarchical recall

    h_f = (2 * h_p * h_r) / (h_p + h_r)  # hierarchical f-score

    return h_f, h_p, h_r





h_f,h_p,h_r=get_hf_scores()



print('\nHierarchical measures:\n \tHPrecision = ' + str(h_p) + '\n \tHRecall = ' + \

              str(h_r) + '\n \tHF-score = ' + str(h_f))



# Hierarchical F score computation ends here.







############################### Text  to classify #####################################

x = input('Enter your arabic text: ')



x_vec = [model.infer_vector(clean_text(x).split())]



dec = clf.predict(x_vec)



if (dec == 1): print('1- ثقافة')

elif  (dec == 2): print('2- ملابس')

elif  (dec == 3): print('3- طعام و شراب')

elif  (dec == 4): print('4- سياحة')

elif  (dec == 5): print('5- تاريخ')

elif  (dec == 6): print('6- أحداث')

elif  (dec == 7): print('7- إختراعات')

elif  (dec == 8): print('8- آثار')

elif  (dec == 9): print('9- رياضيات')

elif  (dec == 10): print('10- جبر')

elif  (dec == 11): print('11- تحليل')

elif  (dec == 12): print('12- هندسة')



