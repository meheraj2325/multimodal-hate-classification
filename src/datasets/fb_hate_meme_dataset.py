from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoImageProcessor, AutoProcessor, AutoTokenizer

from configs import config
from configs.defaults import *
from utils.constants import *
from utils.utils import get_pretrained_model_path

FB_HATE_MEME_DATASET_NAME = "fb_hate_meme"


# Define dataset class
class FBHateMemeDataset(Dataset):
    def __init__(self, args, dataset_type: str = "train"):
        self.args = args
        self.dataset_dir = Path(args.data_dir) / FB_HATE_MEME_DATASET_NAME
        self.dataset_type = dataset_type
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((config.image_width, config.image_height)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
                ),
            ]
        )

        self.load_data()

        if (
            ("clip" in self.args.model_name)
            or ("dual_encoder" in self.args.model_name)
            or ("vilt" in self.args.model_name)
        ):
            self.processor = self.get_processor()
        elif "resnet" in self.args.model_name or "vit" in self.args.model_name:
            pass
        else:
            self.tokenizer = self.get_tokenizer(self.args.tokenizer_name)

    def load_data(self):
        self.data = pd.read_json(f"{self.dataset_dir}/{self.dataset_type}.jsonl")
        # self.data.reset_index(drop=True, inplace=True)
        self.data_len = len(self.data)

    def get_image_processor(self):
        pretrained_model_path = get_pretrained_model_path(self.args.model_name)
        if pretrained_model_path is None:
            pretrained_model_path = get_pretrained_model_path(DEFAULT_IMAGE_PROCESSOR)

        return AutoImageProcessor.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_path
        )

    def get_tokenizer(self):
        pretrained_model_path = get_pretrained_model_path(self.args.model_name)
        if pretrained_model_path is None:
            pretrained_model_path = get_pretrained_model_path(DEFAULT_TOKENIZER)

        return AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_path
        )

    def get_processor(self):
        pretrained_model_path = get_pretrained_model_path(self.args.model_name)
        if pretrained_model_path is None:
            pretrained_model_path = get_pretrained_model_path(DEFAULT_PROCESSOR)

        if "dual_encoder" in self.args.model_name:
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            feature_extractor = ViTFeatureExtractor.from_pretrained(
                "google/vit-base-patch16-224"
            )
            processor = VisionTextDualEncoderProcessor(feature_extractor, tokenizer)

        else:
            processor = AutoProcessor.from_pretrained(
                pretrained_model_name_or_path=pretrained_model_path
            )

        return processor

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
