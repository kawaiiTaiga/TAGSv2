import random
import csv
import re 

def create_contrastive_data(file_path, data_dict):
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile,delimiter='\t')
        writer.writerow(['Anchor', 'Positive', 'Negative'])

        for label, data_list in data_dict.items():
            for i, anchor in enumerate(data_list):
                positive_samples = data_list[:i] + data_list[i+1:]

                for negative_label, negative_data_list in data_dict.items():
                    if negative_label != label:
                        for negative in negative_data_list:
                            writer.writerow([anchor, positive_samples[0], negative])

                            
def preprocess_dataset(dataset):
    labels = dataset['label'] 

    label_indices = {}
    for i, label in enumerate(labels):
        if label not in label_indices:
            label_indices[label] = []
        label_indices[label].append(i)

    label_lists = {}
    for label, indices in label_indices.items():
        if len(indices) > 1:  
            label_lists[label] = []
   
    for label, indices in label_indices.items():
        if label in label_lists: 
            for index in indices:
                data = dataset[index] 
                label_lists[label].append(data['text'])
    return label_lists

def make_fewshot_dataset(label_lists,num_samples):
    fewshot_data = {}
    
    for label,all_data in label_lists.items():
        few_data = random.sample(all_data,num_samples)
        fewshot_data[label] = few_data
    return fewshot_data

def extract_sentences(text):
    sentences = re.split(r'sentence [A-Za-z0-9]+:', text, flags=re.IGNORECASE)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    return sentences