import os
import tensorflow as tf
from tensorflow import keras
import music21
from music21 import *
import sys
import IPython
from IPython.display import Image, Audio
from IPython.display import display

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

us = environment.UserSettings()

us['lilypondPath'] = r"C:\\Program Files (x86)\\LilyPond\usr\\bin\\lilypond.exe"
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
                elif isinstance(element, chord.Chord):
                    notes.append(element)
                

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


#Melody_Snippet = chords_n_notes(Corpus[:100])
#Melody_Snippet = stream.Stream(Corpus[:300])
Melody_Snippet = all_midis[0]
#show(Melody_Snippet)


Melody_Snippet.show("musicxml.png")
Melody_Snippet.show('midi')
