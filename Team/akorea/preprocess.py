import pandas as pd
import json 
import numpy as np

def get_category():
    category_names = ['Backgroud',
                      'UNKNOWN',
                      'General trash',
                      'Paper',
                      'Paper pack',
                      'Metal',
                      'Glass',
                      'Plastic',
                      'Styrofoam',
                      'Plastic bag',
                      'Battery',
                      'Clothing']
    return category_names



def get_category_ann():
        # Read annotations
    dataset_path = '/opt/ml/input/data'
    anns_file_path = dataset_path + '/' + 'train.json'

    with open(anns_file_path, 'r') as f:
        dataset = json.loads(f.read())

    categories = dataset['categories']
    anns = dataset['annotations']
    imgs = dataset['images']
    nr_cats = len(categories)
    nr_annotations = len(anns)
    nr_images = len(imgs)

    cat_names = []
    super_cat_names = []
    super_cat_ids = {}
    super_cat_last_name = ''
    nr_super_cats = 0
    for cat_it in categories:
        cat_names.append(cat_it['name'])
        super_cat_name = cat_it['supercategory']
        # Adding new supercat
        if super_cat_name != super_cat_last_name:
            super_cat_names.append(super_cat_name)
            super_cat_ids[super_cat_name] = nr_super_cats
            super_cat_last_name = super_cat_name
            nr_super_cats += 1

    cat_histogram = np.zeros(nr_cats,dtype=int)
    for ann in anns:
        cat_histogram[ann['category_id']] += 1

    df = pd.DataFrame({'Categories': cat_names, 'Number of annotations': cat_histogram})
    df = df.sort_values('Number of annotations', 0, False)

    sorted_temp_df = df.sort_index()
    sorted_df = pd.DataFrame(["Backgroud"], columns = ["Categories"])
    sorted_df = sorted_df.append(sorted_temp_df, ignore_index=True)
    
    category_names = list(sorted_df.Categories)

    return category_names