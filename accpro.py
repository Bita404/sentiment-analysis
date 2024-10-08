from transformers import pipeline
from sklearn.metrics import accuracy_score, classification_report
from datasets import load_dataset

####### using a pretrained model eng
emotion_classifier = pipeline('text-classification', model='j-hartmann/emotion-english-distilroberta-base', return_all_scores=True)

# ## emotion function ###
def detect_emotion(text):
    results = emotion_classifier(text)
     ########## emotion and score dict extraction
    emotions = [(result['label'], result['score']) for result in results[0]]
    ##### ordering by highest score
    emotions.sort(key=lambda x: x[1], reverse=True)
    return emotions[0][0]  

####### Load dataset

dataset = load_dataset('go_emotions', split='test')
texts = dataset['text']
true_labels = dataset['labels']

#lables 
label_to_emotion = dataset.features['labels'].feature.int2str
true_labels_mapped = [label_to_emotion(label[0]) for label in true_labels] 

####### use the emotions function
predicted_labels = [detect_emotion(text) for text in texts]

######## accuracy
accuracy = accuracy_score(true_labels_mapped, predicted_labels)
print(f"Accuracy: {accuracy:.2f}")

print(classification_report(true_labels_mapped, predicted_labels))
