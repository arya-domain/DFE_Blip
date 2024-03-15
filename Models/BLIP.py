from transformers import AutoProcessor, BlipForConditionalGeneration, AutoConfig, AutoModel
import torch


def BLIP():
    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    # model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")


    config = AutoConfig.from_pretrained("Salesforce/blip-image-captioning-large")
    model = AutoModel.from_config(config)
    return processor, model
