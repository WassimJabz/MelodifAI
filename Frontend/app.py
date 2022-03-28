import io
import os
import sys

from flask import Flask, render_template, request
from PIL import Image


import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import json
from music21 import *
from music21 import converter, chord, note, stream, environment, instrument, duration

app = Flask(__name__)

# model = Model()
# model.load_state_dict(torch.load("weights.pth"))
model = tf.keras.models.load_model('saved_model/Model 5')

#Setting up the corpus
with open('Corpus.json','r') as f:
    Corpus  = json.load(f)
symb = sorted(list(set(Corpus)))
L_corpus = len(Corpus) #length of corpus
L_symb = len(symb) #length of total unique characters
#Building dictionary to access the vocabulary from indices and vice versa
mapping = dict((c, i) for i, c in enumerate(symb))
reverse_mapping = dict((i, c) for i, c in enumerate(symb))

length = 40
features = []
targets = []
for i in range(0, L_corpus - length, 1):
    feature = Corpus[i:i + length]
    target = Corpus[i + length]
    features.append([mapping[j] for j in feature])
    targets.append(mapping[target])
    
    
L_datapoints = len(targets)

X = (np.reshape(features, (L_datapoints, length, 1)))/ float(L_symb)
# one hot encode the output variable
y = keras.utils.to_categorical(targets) 

X_train, X_seed, y_train, y_seed = train_test_split(X, y, test_size=0.2, random_state=42)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/selectionMade", methods=["POST"])
def data():
    """Generate music based on selection"""
    selection = request.data
    seed = X_seed[selection]
    
    Music_notes, Melody = melody_generator(80,seed)
    Melody.write('midi','Melody_Generated.mid')



    return None

def melody_generator(Note_Count, seed):
    """"Melody Generator"""
    
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
        if i!=0 and i%80 == 0:
            seed = X_seed[np.random.randint(0,len(X_seed)-1)]

    #Now, we have music in form or a list of chords and notes and we want to be a midi file.
    Melody = chords_n_notes(Music)
    Melody_midi = stream.Stream(Melody)   
    return Music,Melody_midi

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
            #chord_snip.offset = offset
            Melody.append(chord_snip)
            #offset += float(chord_snip.quarterLength)

        # pattern is a note
        else: 
            dur_and_pitch = i.split(",")
            note_snip = note.Note(int(dur_and_pitch[1]))
            note_snip.duration = duration.Duration(float(dur_and_pitch[0]))
            #note_snip.offset = offset
            Melody.append(note_snip)
            #offset += float(note_snip.quarterLength)
        # increase offset each iteration so that notes do not stack
        
    Melody_midi = stream.Stream(Melody)
    return Melody_midi