import numpy as np
from PIL import Image
from torch.utils import data
import glob
import cv2
import utils
import pandas as pd
import os
from torchvision import transforms as pth_transforms
import torch

# ---------------------------------------------------------------------------------
# Symptom configuration: use only the 5 most frequent findings to simplify training
# ---------------------------------------------------------------------------------

# Original symptoms for data_dir1
SYMPTOMS_V1 = [
    'Infiltration',
    'Effusion',
    'Atelectasis',
    'Nodule',
    'Mass',
    'Pneumothorax',
    'Consolidation',
]

# New symptoms for data_dir2
SYMPTOMS_V2 = [
    'Consolidation',
    'Pneumothorax',
    'Infiltration',
    'Pneumonia',
    'Fibrosis',
    'Effusion'
]

import main_run

parser = main_run.get_args_parser()
args = parser.parse_args()

# Dynamically set NUM_SYMPTOMS based on data_path
if 'data_dir3' in args.data_path:
    # For data_dir3, we need to read the selected symptoms from the CSV
    import pandas as pd
    import os
    data_dir3_csv = os.path.join(args.data_path, 'datafile.csv')
    if os.path.exists(data_dir3_csv):
        # Read CSV and extract unique symptoms
        df_temp = pd.read_csv(data_dir3_csv, nrows=100)  # Read first 100 rows to get symptoms
        all_symptoms = set()
        for _, row in df_temp.iterrows():
            labels = row['Finding Labels'].split('|')
            all_symptoms.update([l for l in labels if l != 'No Finding'])
        # Filter to get the 6 most common symptoms (these should be the selected ones)
        from collections import Counter
        symptom_counter = Counter()
        for _, row in df_temp.iterrows():
            labels = row['Finding Labels'].split('|')
            symptom_counter.update([l for l in labels if l != 'No Finding'])
        SYMPTOMS_V3 = [s[0] for s in symptom_counter.most_common(6)]
        NUM_SYMPTOMS = len(SYMPTOMS_V3)
        SYMPTOM_TO_INDEX = {name: idx for idx, name in enumerate(SYMPTOMS_V3)}
    else:
        # Fallback if CSV doesn't exist yet
        NUM_SYMPTOMS = 6
        SYMPTOM_TO_INDEX = {}
elif 'data_dir2' in args.data_path:
    NUM_SYMPTOMS = len(SYMPTOMS_V2)  # 6 symptoms
    SYMPTOM_TO_INDEX = {name: idx for idx, name in enumerate(SYMPTOMS_V2)}
else:
    NUM_SYMPTOMS = len(SYMPTOMS_V1)  # 5 symptoms
    SYMPTOM_TO_INDEX = {name: idx for idx, name in enumerate(SYMPTOMS_V1)}

class CXR_Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, data_dir, csv_file, transforms=None, mode='train', labeled=False, disease = True): 
        
        'Initialization'

        #if True on disease, that means the dataloader will return just the covid, tb, norm info from test folder
        # if False on disease, that means the dataloader will return symtom test data, the label 
        self.dim = (256, 256)
        self.transforms = transforms
        self.mode = mode
        self.labeled = labeled
        self.disease = disease
        self.data_dir = data_dir
        self.csv_file = csv_file
        self.test_csv_file = 'symptom_test.csv'
        
        # Dynamically select symptoms based on data directory
        if 'data_dir3' in data_dir:
            # For data_dir3, use the dynamically determined symptoms
            if 'SYMPTOMS_V3' in globals():
                self.selected_symptoms = SYMPTOMS_V3
                self.num_symptoms = len(SYMPTOMS_V3)
            else:
                # Fallback - will be populated when CSV is read
                self.selected_symptoms = []
                self.num_symptoms = 6
        elif 'data_dir2' in data_dir:
            self.selected_symptoms = SYMPTOMS_V2
            self.num_symptoms = len(SYMPTOMS_V2)
        else:
            self.selected_symptoms = SYMPTOMS_V1
            self.num_symptoms = len(SYMPTOMS_V1)
        
        self.symptom_to_index = {name: idx for idx, name in enumerate(self.selected_symptoms)}
        
        print(f"Using symptoms for {os.path.basename(data_dir)}: {self.selected_symptoms}")
        
        # SPECIAL CASE FOR GRAD-CAM: Only load the specific image from symptom_test.csv
        if 'gradcam_temp' in data_dir or '/tmp/' in data_dir:
            print("GRAD-CAM MODE: Loading only images specified in symptom_test.csv")
            self.total_images = {}
            self.test_data = pd.read_csv(os.path.join(self.data_dir, self.test_csv_file))
            print(f"Loaded test CSV with {len(self.test_data)} rows")
            
            # Only load the specific images listed in the CSV
            for idx, row in self.test_data.iterrows():
                img_name = row['Image Index']
                img_path = os.path.join(self.data_dir, 'symptom_test', img_name)
                if os.path.exists(img_path):
                    self.total_images[img_path] = {
                        'disease': 0, 
                        'symptoms': self.get_test_symptom_labels(img_path), 
                        'type': 'test'
                    }
                    print(f"  Added image: {img_name}")
                else:
                    print(f"  Warning: Image not found: {img_path}")
            
            self.total_images_list = sorted(self.total_images.keys())
            self.selected_images = self.total_images_list
            self.n_data = len(self.total_images_list)
            print(f"Total images for Grad-CAM: {len(self.total_images_list)}")
            return  # Skip the rest of initialization
        
        if args.total_folds == 0:
            self.total_folds = ['labeled']
            self.pseudo_folds = []
        elif args.total_folds == 1:
            self.total_folds = ['labeled']
            self.pseudo_folds = ['fold_0']
        elif args.total_folds == 2:
            self.total_folds = ['labeled']
            self.pseudo_folds = ['fold_0', 'fold_1']
        elif args.total_folds == 3:
            self.total_folds = ['labeled']
            self.pseudo_folds = ['fold_0', 'fold_1', 'fold_2']

        self.test_fold = ['test']

        self.total_images = {}
        print(f"Initializing dataset with mode='{self.mode}', labeled={self.labeled}, disease={self.disease}")

        # Load data from CSV
        self.data = pd.read_csv(os.path.join(data_dir, csv_file))
        print("CSV file loaded. Number of rows:", len(self.data))

        self.test_data =  pd.read_csv(os.path.join(data_dir, self.test_csv_file))
        
        # Load images from fold directories and Images directory
        self.load_images()

        self.total_images_list = sorted(self.total_images.keys())
        self.selected_images = self.total_images_list

        print('A total of %d image data were generated.' % len(self.selected_images))
        self.n_data = len(self.selected_images)

    def load_images(self):
        'Load images and labels from fold directories and combine with CSV data'

        def add_images_from_fold(fold, label_type):
            png_lists = glob.glob(os.path.join(self.data_dir, fold, '**/*.png'), recursive=True)
            jpg_lists = glob.glob(os.path.join(self.data_dir, fold, '**/*.jpg'), recursive=True)
            for img_path in png_lists + jpg_lists:
                # Use symptom labels from CSV when available; fall back to zeros
                sympt = self.get_symptom_labels(img_path)
                if 'Normal' in img_path:
                    self.total_images[img_path] = {'disease': 0, 'symptoms': sympt, 'type': label_type}
                elif 'Tuberculosis' in img_path:
                    self.total_images[img_path] = {'disease': 1, 'symptoms': sympt, 'type': label_type}
                elif 'Covid' in img_path:
                    self.total_images[img_path] = {'disease': 2, 'symptoms': sympt, 'type': label_type}
                else:
                                    self.total_images[img_path] = {'disease': 9999, 'symptoms': sympt, 'type': label_type}



        image_files = glob.glob(os.path.join(self.data_dir, 'Images', '*.png')) + \
                    glob.glob(os.path.join(self.data_dir, 'Images', '*.jpg'))
        num_images = len(image_files)
        # Use 30% of Images directory for symptom supervision when labeled=True
        labeled_cutoff = int(0.3 * num_images)
        fold_0_cutoff = labeled_cutoff + int(0.3 * num_images)
        fold_1_cutoff = fold_0_cutoff + int(0.3 * num_images)

        if self.mode == 'train':
            if self.labeled:
                # Add images from 'labeled' fold
                add_images_from_fold('labeled', 'label')
                # Add first 10% of images from 'Images' directory
                for img_path in image_files[:labeled_cutoff]:
                    self.total_images[img_path] = {'disease': 0, 'symptoms': self.get_symptom_labels(img_path), 'type': 'label'}
            else:
                # Add images from the relevant folds
                if args.total_folds >= 1:
                    add_images_from_fold('fold_0', 'pseudo')
                    for img_path in image_files[labeled_cutoff:fold_0_cutoff]:
                        self.total_images[img_path] = {'disease': 0, 'symptoms': self.get_symptom_labels(img_path), 'type': 'fold_0'}
                if args.total_folds >= 2:
                    add_images_from_fold('fold_1', 'pseudo')
                    for img_path in image_files[fold_0_cutoff:fold_1_cutoff]:
                        self.total_images[img_path] = {'disease': 0, 'symptoms': self.get_symptom_labels(img_path), 'type': 'fold_1'}
                if args.total_folds == 3:
                    add_images_from_fold('fold_2', 'pseudo')
                    for img_path in image_files[fold_1_cutoff:]:
                        self.total_images[img_path] = {'disease': 0, 'symptoms': self.get_symptom_labels(img_path), 'type': 'fold_2'}

                # If total_folds == 0 no fold_0 etc. were added; fall back to use full Images directory
                if args.total_folds == 0 and len(self.total_images) == 0:
                    for img_path in image_files:
                        self.total_images[img_path] = {
                            'disease': 0,
                            'symptoms': self.get_symptom_labels(img_path),
                            'type': 'pseudo'
                        }
        elif self.mode == 'test' :
            if self.disease == True:
                add_images_from_fold('test', 'test')
            if self.disease == False:
                test_symptom_files = glob.glob(os.path.join(self.data_dir, 'symptom_test', '*.png')) + \
                                     glob.glob(os.path.join(self.data_dir, 'symptom_test', '*.jpg'))
                print(f"Loading {len(test_symptom_files)} test images from symptom_test directory")
                
                for img_path in test_symptom_files:
                    self.total_images[img_path] = {'disease': 0, 'symptoms': self.get_test_symptom_labels(img_path), 'type': 'test'}


    def get_test_symptom_labels(self, img_path):
            filename = os.path.basename(img_path)
            row = self.test_data[self.test_data['Image Index'] == filename]
            symptom_test_labels = np.zeros(self.num_symptoms)
            if not row.empty:
                label_str = row.iloc[0]['Finding Labels']
                labels = label_str.split('|')
                for label in labels:
                    if label in self.symptom_to_index:
                        symptom_test_labels[self.symptom_to_index[label]] = 1

            return symptom_test_labels


    def get_symptom_labels(self, img_path):
        filename = os.path.basename(img_path)
        row = self.data[self.data['Image Index'] == filename]
        symptom_labels = np.zeros(self.num_symptoms)
        if not row.empty:
            label_str = row.iloc[0]['Finding Labels']
            labels = label_str.split('|')
            for label in labels:
                if label in self.symptom_to_index:
                    symptom_labels[self.symptom_to_index[label]] = 1

        return symptom_labels

    def __len__(self):
        'Denotes the total number of samples'
        return self.n_data

    def __getitem__(self, index):
        'Generates one sample of data'

        img_path = self.total_images_list[index]
        image = cv2.imread(img_path, 1)
        assert image is not None, f"Failed to read image {img_path}"
        image = cv2.resize(image, dsize=(256, 256), interpolation=cv2.INTER_LINEAR)  # type: ignore[arg-type]
        image = Image.fromarray(image)

        # Apply DINO augmentation
        if self.transforms is not None:
            image = self.transforms(image)
        else:
            image = pth_transforms.Compose(
                [utils.GaussianBlurInference(),
                 pth_transforms.ToTensor(),
                 pth_transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                           std=(0.229, 0.224, 0.225))])(image)

        disease_label = self.total_images[img_path]['disease']
        symptom_labels = self.total_images[img_path]['symptoms']

        # print("Symptoms Labels")
        # print(symptom_labels)
        # print("Disease Label")
        # print(disease_label)
        # print("Img_Path")
        #print(self.total_images[img_path])



        symptom_labels = torch.from_numpy(symptom_labels.astype(np.float32))

        if self.mode == 'test' and self.disease == True:
            return image, disease_label, img_path
        elif self.mode == 'test' and self.disease == False:
            return image, symptom_labels


        return image, int(disease_label), symptom_labels