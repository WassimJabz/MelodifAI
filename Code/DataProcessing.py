import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adamax
import pandas as pd 
import seaborn as sns
import music21
from music21 import *
from music21 import converter, chord, note, stream, environment, instrument
import sys
import IPython
from IPython.display import Image, Audio
import matplotlib.pyplot as plt 
from IPython.display import display
import numpy as np 
#import pandas as pd 
from collections import Counter
from sklearn.model_selection import train_test_split




import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

us = environment.UserSettings()

# us['lilypondPath'] = r"C:\\Program Files (x86)\\LilyPond\usr\\bin\\lilypond.exe"
us['musicxmlPath'] = r"C:\\Program Files\\MuseScore 3\\bin\\MuseScore3.exe"
us['musescoreDirectPNGPath'] = r"C:\\Program Files\\MuseScore 3\\bin\\MuseScore3.exe"

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
                    notes.append(element)
                    #notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append(element)
                    #notes.append(".".join(str(n) for n in element.normalOrder))
                #elif isinstance(element, note.Rest):
                 #   notes.append(element)
            
                
                

    return notes

Corpus = extract_notes(all_midis)
print("Total notes in all the Chopin midis in the dataset:", len(Corpus))

def show(music):
    """Show score sheet"""
    display(Image(str(music.write("lily.png"))))
    
def chords_n_notes(Snippet):
    """show notes and chords"""
    Melody = []
    offset = 0 #Incremental
    for i in Snippet:
        #If it is chord
        if ("." in i or i.isdigit()):
            chord_notes = i.split(".") #Seperating the notes in chord
            notes = [] 
            for j in chord_notes:
                inst_note=int(j)
                note_snip = note.Note(inst_note)           
                notes.append(note_snip)
                chord_snip = chord.Chord(notes)
                chord_snip.offset = offset
                Melody.append(chord_snip)
                
        # pattern is a note
        else: 
            note_snip = note.Note(i)
            note_snip.offset = offset
            Melody.append(note_snip)
        # increase offset each iteration so that notes do not stack
        offset += 1
    Melody_midi = stream.Stream(Melody)
    return Melody_midi


def chords_n_notes_duration(Snippet):
    """show notes and chords"""
    Melody = stream.Stream()
    #offset = 0.0 #Incremental
    for element in Snippet:
        #If it is chord
        if isinstance(element, chord.Chord):
            note_snip = chord.Chord(element.pitches)
            note_snip.duration = element.duration
            note_snip.offset = element.offset
            Melody.insert(note_snip)
            #offset += element.duration.quarterLength
        # pattern is a note
        elif isinstance(element, note.Note):
            note_snip = note.Note(element.pitch)
            note_snip.duration = element.duration
            note_snip.offset = element.offset
            Melody.insert(note_snip)
            #offset += element.duration.quarterLength
        else:
            note_snip = note.Rest()
            note_snip.duration = element.duration
            note_snip.offset = element.offset
            Melody.insert(note_snip)
            #offset += element.duration.quarterLength
        # increase offset each iteration so that notes do not stack
        #offset +=1
    #Melody_midi = stream.Stream(Melody)
    Melody_midi = Melody
    return Melody_midi


Melody_Snippet = chords_n_notes_duration(Corpus[:300])

Melody_Snippet.show()
#Melody_Snippet = stream.Stream(Corpus[:300])
#Melody_Snippet = all_midis[0]
#show(Melody_Snippet)