import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, ViTModel


class GuidedAttentionModel(pl.LightningModule):
    def __init__(self, num_classes, train_backbone=True, lr=1e-4):
        super().__init__()
        self.lr = lr
        self.num_classes = num_classes
        self.train_backbone = train_backbone

        # Load pre-trained models
        self.vit_original = ViTModel.from_pretrained(
            "google/vit-base-patch16-224-in21k"
        )
        self.vit_processed = ViTModel.from_pretrained(
            "google/vit-base-patch16-224-in21k"
        )
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        # Freeze or unfreeze backbone parameters based on train_backbone flag
        if not self.train_backbone:
            for param in self.vit_original.parameters():
                param.requires_grad = False
            for param in self.vit_processed.parameters():
                param.requires_grad = False
            for param in self.bert.parameters():
                param.requires_grad = False

        # Define attention and MLP layers
        self.attention1 = nn.MultiheadAttention(embed_dim=768, num_heads=8)
        self.attention2 = nn.MultiheadAttention(embed_dim=768, num_heads=8)
        self.mlp1 = nn.Linear(768, 128)
        self.mlp2 = nn.Linear(768, 128)
        self.mlp3 = nn.Linear(256, num_classes)

    def forward(
        self, original_image, extracted_image, text_input_ids, text_attention_mask
    ):
        original_features = self.vit_original(
            pixel_values=original_image
        ).last_hidden_state
        extracted_features = self.vit_processed(
            pixel_values=extracted_image
        ).last_hidden_state
        bert_outputs = self.bert(
            input_ids=text_input_ids, attention_mask=text_attention_mask
        )

        original_cls = original_features[:, 0, :]
        extracted_cls = extracted_features[:, 0, :]
        bert_cls = bert_outputs.pooler_output

        attn_output1, _ = self.attention1(
            query=extracted_cls.unsqueeze(0),
            key=original_cls,
            value=original_cls,
        )
        attn_output2, _ = self.attention2(
            query=bert_cls.unsqueeze(0), key=original_cls, value=original_cls
        )

        mlp_output1 = F.relu(self.mlp1(attn_output1.squeeze(0)))
        mlp_output2 = F.relu(self.mlp2(attn_output2.squeeze(0)))

        fusion_output = torch.cat((mlp_output1, mlp_output2), dim=1)
        fusion_output = self.mlp3(fusion_output)

        return fusion_output

    def training_step(self, batch, batch_idx):
        original_images = batch["original_image"]
        extracted_images = batch["extracted_image"]
        text_input_ids = batch["text_input_ids"]
        text_attention_mask = batch["text_attention_mask"]
        labels = batch["label"]
        outputs = self(
            original_images, extracted_images, text_input_ids, text_attention_mask
        )
        loss = F.cross_entropy(outputs, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        original_images = batch["original_image"]
        extracted_images = batch["extracted_image"]
        text_input_ids = batch["text_input_ids"]
        text_attention_mask = batch["text_attention_mask"]
        labels = batch["label"]
        outputs = self(
            original_images, extracted_images, text_input_ids, text_attention_mask
        )
        loss = F.cross_entropy(outputs, labels)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", patience=3, factor=0.5, verbose=True
            ),
            "monitor": "val_loss",
        }
