# MelodifAI

For more information about this project please visit our [Devpost](https://devpost.com/software/melodifai).

# Requirements
- Python 3.x
- Tensorflow
- Flask
- Music21
- pandas
- seaborn
- IPython
- matplotlib

(Optional) MuseScore 3 to view MIDI files

# Training

MelodifAIV2.0.py will train a new model and save the weights as 'my_model'. Rename this file to avoid it being overwritten on subsequent trainings.

# Prediction

To run the local webapp, run app.py and access the interface on http:localhost:5000. The model that will be used can be changed in model.py in the __init__() method.




