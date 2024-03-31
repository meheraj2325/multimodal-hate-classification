import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from transformers import BertTokenizer, ViTFeatureExtractor


# Define dataset class
class FBHateMemeDataset(Dataset):
    def __init__(self, dataframe, image_size=(224, 224)):
        self.data = dataframe
        self.image_size = image_size
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.vit_feature_extractor = ViTFeatureExtractor.from_pretrained(
            "google/vit-base-patch16-224-in21k"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.iloc[idx]["original_image_paths"]
        extracted_image_path = self.data.iloc[idx]["extracted_image_paths"]
        text = self.data.iloc[idx]["text"]
        label = self.data.iloc[idx]["label"]

        # Load and preprocess images
        original_image = Image.open(image_path).convert("RGB")
        original_image = self.image_transform(original_image)

        extracted_image = Image.open(extracted_image_path).convert("RGB")
        extracted_image = self.image_transform(extracted_image)

        # Tokenize text
        text_encoding = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128,
        )

        return {
            "original_image": original_image,
            "extracted_image": extracted_image,
            "text_input_ids": text_encoding["input_ids"].squeeze(0),
            "text_attention_mask": text_encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label),
        }
