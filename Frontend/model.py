import io
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import json
import music21
from music21 import *
from music21 import converter, chord, note, stream, environment, instrument, duration
from music21.tempo import MetronomeMark


class Model:
    """
    
    """

    def __init__(self):
     path = os.getcwd()
     
     filepath = os.path.join(path[:len(path)-9],"saved_model\\Model 4")
     self.model = tf.keras.models.load_model(filepath)   

     with open(os.path.join(path[:len(path)-9],'Corpus.json'),'r') as f:
         self.Corpus  = json.load(f)

     
     
     


     symb = sorted(list(set(self.Corpus)))
     L_corpus = len(self.Corpus) #length of corpus
     self.L_symb = len(symb) #length of total unique characters
     #Building dictionary to access the vocabulary from indices and vice versa
     mapping = dict((c, i) for i, c in enumerate(symb))
     self.reverse_mapping = dict((i, c) for i, c in enumerate(symb))

     self.length = 40
     features = []
     targets = []
     for i in range(0, L_corpus - self.length, 1):
         feature = self.Corpus[i:i + self.length]
         target = self.Corpus[i + self.length]
         features.append([mapping[j] for j in feature])
         targets.append(mapping[target])
     
     
        
     L_datapoints = len(targets)

     
     X = (np.reshape(features, (L_datapoints, self.length, 1)))/ float(self.L_symb)
     # one hot encode the output variable
     y = keras.utils.to_categorical(targets) 

     X_train, X_seedOld, y_train, y_seed = train_test_split(X, y, test_size=0.2, random_state=42)

     
        

    def melody_generator(self, Note_Count):
     """"Melody Generator"""
     seed = self.X_seed[np.random.randint(0,len(self.X_seed)-1)]
     #seed = self.seed1[:40]
     Music = ""
     Notes_Generated= []

     for i in range(Note_Count):
         seed = seed.reshape(1,self.length,1)
         prediction = self.model.predict(seed, verbose=0)[0]
         #prediction = np.log(prediction) / 1.0 #diversity
         exp_preds = np.exp(prediction)
         prediction = exp_preds / np.sum(exp_preds)
         index = np.argmax(prediction)
         index_N = index/ float(self.L_symb)   
         Notes_Generated.append(index)
         Music = [self.reverse_mapping[char] for char in Notes_Generated]
         seed = np.insert(seed[0],len(seed[0]),index_N)
         seed = seed[1:]
         if i!=0 and i%40 == 0:
             seed = self.X_seed[np.random.randint(0,len(self.X_seed)-1)]

     #Now, we have music in form or a list of chords and notes and we want to be a midi file.
     Melody = self.chords_n_notes(Music)
     Melody_midi = stream.Stream(Melody)
     Melody_midi.insert(MetronomeMark(number=50))
      
     return Music,Melody_midi
    

    def chords_n_notes(self, Snippet):
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
    
    def predict(self, request):
        path = os.getcwd()
        file_seed = 'seed' + str(int(request.data)) + '.json' 
        with open(os.path.join(path[:len(path)-9],file_seed),'r') as f:
         self.seed  = json.load(f)
        
        seed_symb = sorted(list(set(self.Corpus)))
        self.seed_L_symb = len(seed_symb) #length of total unique characters
        #Building dictionary to access the vocabulary from indices and vice versa
        seed_mapping = dict((c, i) for i, c in enumerate(seed_symb))


        self.seed_features = []
        seed_targets = []
        for i in range(0, len(self.seed) - self.length, 1):
            seed_feature = self.seed[i:i + self.length]
            seed_target = self.seed[i + self.length]
            self.seed_features.append([seed_mapping[j] for j in seed_feature])
            seed_targets.append(seed_mapping[seed_target])
        
        self.X_seed = (np.reshape(self.seed_features, (len(seed_targets), self.length, 1)))/ float(self.L_symb)



        print(f"loaded {request}")

        Music_notes, Melody = self.melody_generator(80)
        Melody.write('midi','Melody_Generated.mid')
        return 'Melody_Generated.mid'
        

