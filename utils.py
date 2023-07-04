import random

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
