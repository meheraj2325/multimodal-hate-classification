from typing import Optional

from configs.pretrained_model_paths import *


def get_pretrained_model_path(model_name: str) -> Optional[str]:
    model_prefix_to_pretrained_path = {
        "bert": BERT_PRETRAINED_PATH,
        "clip": CLIP_PRETRAINED_PATH,
        "lxmert": LXMERT_PRETRAINED_PATH,
        "resnet": ResNet_PRETRAINED_PATH,
        "roberta": RoBERTa_PRETRAINED_PATH,
        "electra": Electra_PRETRAINED_PATH,
        "vilt": ViLT_PRETRAINED_PATH,
        "vit": ViT_PRETRAINED_PATH,
    }

    model_name = model_name.lower()
    for model_prefix, pretrained_path in model_prefix_to_pretrained_path.items():
        if model_prefix in model_name:
            return pretrained_path

    return None
