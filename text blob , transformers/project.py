import transformers
from transformers import pipeline
from datasets import load_dataset

####### using a pretrained model eng

emotion_classifier = pipeline('text-classification', model='j-hartmann/emotion-english-distilroberta-base', return_all_scores=True)
#emotion_classifier = pipeline('text-classification', model='HooshvareLab/bert-fa-base-uncased', return_all_scores=True)


### emotion function ###

def detect_emotion(text):
    results = emotion_classifier(text)
    
    ########## emotion and score dict extraction
    emotions = [(result['label'], result['score']) for result in results[0]]
    
    ##### ordering by highest score
    emotions.sort(key=lambda x: x[1], reverse=True)
    return emotions


#text_sample= [
    
    #"jojo is an amazing anime I recommend it to everyone",
    # "that noodle was awful",
    # "the exam was normal"
#]

dataset = load_dataset('go_emotions', split='train')
texts = dataset['text']

for text in texts:
    emotions = detect_emotion(text)
    print(f"Text: {text}")
    for emotion, score in emotions:
        print(f"  {emotion}: {score:.2f}")
    print()

