from __future__ import annotations


EMOTION_LABELS = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral",
}

# FERPlus 10-annotator columns (fer2013_new.csv) -> 8 semantic emotions
FERPLUS_EMOTIONS = [
    "neutral",
    "happiness",
    "surprise",
    "sadness",
    "anger",
    "disgust",
    "fear",
    "contempt",
]

# Map FERPlus 8-emotion probabilities down to the 7-class FER-2013 schema
FERPLUS_TO_7 = {
    "neutral": 6,
    "happiness": 3,
    "surprise": 5,
    "sadness": 4,
    "anger": 0,
    "disgust": 1,
    "fear": 2,
    "contempt": 6,
}
