import pickle
from nltk.corpus import sentiwordnet as swn
import nltk
from keras import Sequential
from keras.layers import Dense
from nltk.stem.lancaster import LancasterStemmer
import numpy
import tensorflow
import random
import json
import keras
import numpy as np
import pyttsx3
import pandas as pd

like_mov, dislike_mov = {}, {}

"""
Project Phase II: Chatbot 

Author - Lathwahandi Diyohan De Silva

"""

# preprocessing for the movie database
stemmer = LancasterStemmer()
metadata = pd.read_csv('movies_metadata.csv', low_memory=False)
metadata.drop_duplicates(subset="original_title",
                         keep=False, inplace=True)

meta = metadata.set_index("original_title")

user = ''

# preprocess for train data
with open("chatdata.json") as file:
    data = json.load(file)

    words = []
    label = []
    feature = []
    target = []
    for predict in data['intents']:

        for pattern in predict['patterns']:
            test = nltk.word_tokenize(pattern)
            words.extend(test)
            feature.append(test)
            target.append(predict["tag"])

            if predict['tag'] not in label:
                label.append(predict["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w not in "?"]
    words = sorted(list(set(words)))
    label = sorted(label)

    train = []
    output = []

    out_null = [0 for _ in range(len(label))]

    for x, doc in enumerate(feature):
        bag = []
        test = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in test:
                bag.append(1)
            else:
                bag.append(0)

        output_set = out_null[:]
        output_set[label.index(target[x])] = 1

        train.append(bag)
        output.append(output_set)

    training = numpy.array(train)
    # print(training)
    output = numpy.array(output)
    # print(output)

    # print("This is the training shape size ", )
    # print(training[0:1].shape)

    # print("Output size should be number of classes:", len(output[0]))
    # print("input shape shoulld be number of words in the doc", len(training[0]))
# Neural Network for the Dialog classification
model = keras.Sequential([
    keras.layers.Dense(len(output[0]), input_shape=(len(training[0]),), activation='relu'),
    # keras.layers.Flatten(),
    keras.layers.Dense(8),
    keras.layers.Dense(8),
    keras.layers.Dense(len(output[0]), activation="softmax")
])  #

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(training, output,
                    batch_size=8,
                    epochs=2000,
                    verbose=1,
                    validation_split=0.1)
print(model.summary())
model.fit(training, output, epochs=2000, batch_size=8)
model.save("model.tflearn")

"""
@parameter:
s= string containing movie title

@return 


method finds the corresponding plot details for the given movie title

"""


def checkmov(s):
    try:
        global meta
        test = meta.loc[s, 'overview']
        print(test)


    except:
        print('Cannot find the movie, please try an different movie')

    return


"""
@parameter:
s= input dialog string
words = tokenized word list of training data
@return 


method compares the input dialog string against the training data and create an bag of words vector/array

"""


def bagwords(s, words):
    bags = np.array([0 for _ in range(len(words))])

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    for tok in s_words:
        for i, w in enumerate(words):
            if w == tok:
                bags[i] = 1

    bags = bags.reshape(1, len(training[0]))
    return bags


# sen. analysis for movie like/dislikes
"""
@parameter:
x= string containing user review of the movie
y= title name of the movie
z= user name
@return 


method perform sentinel analysis on input: user review string and append the movie title to user name based on pos/neg. intent score

"""


def likemovie(x, y, z):
    pos = 0
    neg = 0
    tokens = nltk.word_tokenize(x)
    for token in tokens:
        syn_list = list(swn.senti_synsets(token))
        if syn_list:
            syn = syn_list[0]
            neg += syn.neg_score()
            pos += syn.pos_score()

    if neg > pos:
        if len(z) == 0:
            z = 'Unknown user'
            dislike_mov.setdefault(z, []).append(y)

        else:
            dislike_mov.setdefault(z, []).append(y)

    elif pos > neg:
        if len(z) == 0:
            z = 'Unknown user'
            like_mov.setdefault(z, []).append(y)
        else:
            like_mov.setdefault(z, []).append(y)
    return


"""
@parameter:
x= user name
y= dict containing liked movies
z= dict containing dislike movies
@return 


method takes the above parameters and print user's movie preferences 

"""


def self(name, like, dislike):
    if len(name) == 0:
        print("Please try again after talking to me more ")

    else:
        for name, v in like.items():
            if name in like_mov:
                for a in range(len(v)):
                    print("User: " + name + "| Like Movies: " + v[a])

            if name in dislike_mov:
                for name, k in dislike.items():
                    for g in range(len(k)):
                        print("User: " + name + "| Dislike Movies: " + k[g])

    return


"""
@parameter:

@:return

primary function that acts as the framework for the chatbot 

"""


def chatbot():
    global user, like_mov, dislike_mov
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)
    engine.say("You may speak with the bot now. (Type quit to Stop)")
    engine.runAndWait()
    print("You may speak with the bot now. (Type quit to Stop)")
    while True:
        print("Ask me something!")
        inp = input("You:")
        if inp.lower() == "quit":
            break

        results = model.predict(bagwords(inp, words))[0]
        resultprob = numpy.argmax(results)  # Take the label with highest probability
        tag = label[resultprob]

        if results[resultprob] > 0.8:  # error threshold check

            for r in data["intents"]:
                if r["tag"] == tag:
                    if tag == "name":
                        response = r["responses"]
                        res = random.choices(response)
                        print(res)
                        engine.say(res)
                        engine.runAndWait()
                        user = input("Type your name:")
                        print("Nice to meet you", user)
                        engine.say("Nice to meet you" + user)
                        engine.runAndWait()

                    elif tag == "search":
                        response = r["responses"]
                        res = random.choices(response)
                        print(res)
                        engine.say(res)
                        engine.runAndWait()
                        names = input("Type the name of the movie:")
                        print("Here is a brief overview of the movie")
                        engine.say("Here is a brief overview of the movie")
                        engine.runAndWait()
                        checkmov(names)
                        review = input("So are you interested in this movie?")
                        likemovie(review, names, user)

                    elif tag == "self":
                        response = r["responses"]
                        res = random.choices(response)
                        print(res)
                        engine.say(res)
                        engine.runAndWait()
                        self(user, like_mov, dislike_mov)

                    else:
                        response = r["responses"]
                        res = random.choices(response)
                        print(res)
                        engine.say(res)
                        engine.runAndWait()
        else:
            print("I don't quite understand what you mean. Please try typing again")
            engine.say("I don't quite understand what you mean. Please try typing again")
            engine.runAndWait()


chatbot()
