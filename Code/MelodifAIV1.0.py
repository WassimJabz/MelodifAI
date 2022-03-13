from fractions import Fraction
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adamax

from sklearn.model_selection import train_test_split

import pandas as pd 
import seaborn as sns

import music21
from music21 import *
from music21 import converter, chord, note, stream, environment, instrument

import sys
import IPython
from IPython.display import Image, Audio
from IPython.display import display
import matplotlib.pyplot as plt 
import numpy as np 
from collections import Counter


import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")


#Set Environment Path Variables
us = environment.UserSettings()

# us['lilypondPath'] = r"C:\\Program Files (x86)\\LilyPond\usr\\bin\\lilypond.exe"
us['musicxmlPath'] = r"C:\\Program Files\\MuseScore 3\\bin\\MuseScore3.exe"
us['musescoreDirectPNGPath'] = r"C:\\Program Files\\MuseScore 3\\bin\\MuseScore3.exe"


#Load all midi files in Training folder
path = os.getcwd()
filepath = os.path.join(path,"dataset\\chopinTrain\\")
all_midis = []
for i in os.listdir(filepath):
    if i.endswith(".mid"):
        tr = filepath+i
        midi = converter.parse(tr)
        all_midis.append(midi)
        print(f"added {i}")



def extract_notes(file):
    """extract notes and chords"""
    notes = []
    pick = None
    for j in file:

        songs = instrument.partitionByInstrument(j)

        for part in songs.parts:
            
            pick = part.recurse()
            for element in pick:
                if isinstance(element, note.Note):
                    #notes.append(element)
                    element.dots = 0
                    notes.append(str(float(element.quarterLength)) + "," + str(element.pitch))
                elif isinstance(element, chord.Chord):
                    #notes.append(element)
                    element.dots = 0
                    chordStr = "$" + str(float(element.quarterLength)) + "," + ".".join(str(n) for n in element.normalOrder)
                    notes.append(chordStr)
                elif isinstance(element, note.Rest):
                    restStr = "&" + str(float(element.quarterLength))
                    notes.append(restStr)
                 #   notes.append(element)

            
                
                

    return notes


Corpus = extract_notes(all_midis)
print("Total notes in all the Chopin midis in the dataset:", len(Corpus))

    
def chords_n_notes(Snippet):
    """show notes and chords"""
    Melody = []
    offset = 0.0
    for i in Snippet:
        #If it is chord
        if ("$" in i ):
            i = i[1:]
            dur_and_chord = i.split(",")
            chord_notes = dur_and_chord[1].split(".") #Seperating the notes in chord
            notes = [] 
            for j in chord_notes:
                inst_note=int(j)
                note_snip = note.Note(inst_note)
                note_snip.duration = duration.Duration(float(dur_and_chord[0]))
                notes.append(note_snip)
            chord_snip = chord.Chord(notes)
            chord_snip.duration = duration.Duration(float(dur_and_chord[0]))
            chord_snip.offset = offset
            Melody.append(chord_snip)
            offset += float(chord_snip.quarterLength)
        
        elif("&" in  i):
            i = i[1:]
            rest= note.Rest()
            rest.duration = duration.Duration(float(i))
            rest.offset = offset
            Melody.append(rest)
            offset += float(rest.quarterLength)

        # pattern is a note
        else: 
            dur_and_pitch = i.split(",")
            note_snip = note.Note(dur_and_pitch[1])
            note_snip.duration = duration.Duration(float(dur_and_pitch[0]))
            note_snip.offset = offset
            Melody.append(note_snip)
            offset += float(note_snip.quarterLength)
        # increase offset each iteration so that notes do not stack
        
    Melody_midi = stream.Stream(Melody)
    return Melody_midi



#Melody_Snippet = chords_n_notes(Corpus[:300])

#Melody_Snippet = stream.Stream(Corpus[:300])
#Melody_Snippet = all_midis[0]
#Melody_Snippet.show()
count_num = Counter(Corpus)
print("Total unique notes in the Corpus:", len(count_num))

Notes = list(count_num.keys())
Recurrence = list(count_num.values())

def Average(lst):
    return sum(lst) / len(lst)
print("Average recurrenc for a note in Corpus:", Average(Recurrence))
print("Most frequent note in Corpus appeared:", max(Recurrence), "times")
print("Least frequent note in Corpus appeared:", min(Recurrence), "time")

plt.figure(figsize=(18,3),facecolor="#97BACB")
bins = np.arange(0,(max(Recurrence)), 10) 
plt.hist(Recurrence, bins=bins, color="#97BACB")
plt.axvline(x=100,color="#DBACC1")
plt.title("Frequency Distribution Of Notes In The Corpus")
plt.xlabel("Frequency Of Chords in Corpus")
plt.ylabel("Number Of Chords")
plt.show()

rare_note = []
for index, (key, value) in enumerate(count_num.items()):
    if value < 30:
        m =  key
        rare_note.append(m)
    elif value == "&0.0625":
        m = key
        rare_note.append(m)
        print(f"note found {value}")
    
        
print("Total number of notes that occur less than 20 times:", len(rare_note))

for element in Corpus:
    if element in rare_note:
        Corpus.remove(element)

print("Length of Corpus after elemination the rare notes:", len(Corpus))

count_num2 = Counter(Corpus)
print("Total unique notes in the Corpus:", len(count_num2))

# Storing all the unique characters present in my corpus to bult a mapping dic. 
symb = sorted(list(set(Corpus)))
L_corpus = len(Corpus) #length of corpus
L_symb = len(symb) #length of total unique characters

#Building dictionary to access the vocabulary from indices and vice versa
mapping = dict((c, i) for i, c in enumerate(symb))
reverse_mapping = dict((i, c) for i, c in enumerate(symb))

print("Total number of characters:", L_corpus)
print("Number of unique characters:", L_symb)

#Splitting the Corpus in equal length of strings and output target
length = 40
features = []
targets = []
for i in range(0, L_corpus - length, 1):
    feature = Corpus[i:i + length]
    target = Corpus[i + length]
    features.append([mapping[j] for j in feature])
    targets.append(mapping[target])
    
    
L_datapoints = len(targets)

# reshape X and normalize
X = (np.reshape(features, (L_datapoints, length, 1)))/ float(L_symb)
# one hot encode the output variable
y = keras.utils.to_categorical(targets) 

X_train, X_seed, y_train, y_seed = train_test_split(X, y, test_size=0.2, random_state=42)



print("Total number of sequences in the Corpus:", L_datapoints)


#Model Building


#Initialising the Model
model = Sequential()

#Adding layers
model.add(LSTM(1024, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(512))
model.add(Dense(512))
model.add(Dropout(0.1))
model.add(Dense(y.shape[1], activation='softmax'))

#Compiling the model for training  
opt = Adamax(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=opt)

#Model's Summary               
model.summary()

#Training the Model
history = model.fit(X_train, y_train, batch_size=256, epochs=200)

model.save('saved_model/my_model')

#Plotting the learnings 
history_df = pd.DataFrame(history.history)
fig = plt.figure(figsize=(15,4), facecolor="#97BACB")
fig.suptitle("Learning Plot of Model for Loss")
pl=sns.lineplot(data=history_df["loss"],color="#444160")
pl.set(ylabel ="Training Loss")
pl.set(xlabel ="Epochs")
plt.show()


def melody_Generator(Note_Count):
    """"Melody Generator"""
    seed = X_seed[np.random.randint(0,len(X_seed)-1)]
    Music = ""
    Notes_Generated=[]
    for i in range(Note_Count):
        seed = seed.reshape(1,length,1)
        prediction = model.predict(seed, verbose=0)[0]
        prediction = np.log(prediction) / 1.0 #diversity
        exp_preds = np.exp(prediction)
        prediction = exp_preds / np.sum(exp_preds)
        index = np.argmax(prediction)
        index_N = index/ float(L_symb)   
        Notes_Generated.append(index)
        Music = [reverse_mapping[char] for char in Notes_Generated]
        seed = np.insert(seed[0],len(seed[0]),index_N)
        seed = seed[1:]
    #Now, we have music in form or a list of chords and notes and we want to be a midi file.
    Melody = chords_n_notes(Music)
    Melody_midi = stream.Stream(Melody)   
    return Music,Melody_midi


#getting the Notes and Melody created by the model
Music_notes, Melody = melody_Generator(400)
Melody.write('midi','Melody_Generated.mid')
Melody.show()


#To save the generated melody


#to play audio or corpus
#IPython.display.Audio("../input/music-generated-lstm/Melody_Generated 2.wav")

#to play audio or corpus
#IPython.display.Audio("../input/music-generated-lstm/Melody_Generated_1.wav")

