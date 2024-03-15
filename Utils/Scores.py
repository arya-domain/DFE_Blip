import numpy as np
import torch
from tqdm import tqdm


def DFE_score(labels, predictions):
    original_label = np.argmax(labels, axis=1)
    predicted_label = np.argmax(predictions, axis=1)
    matching_percentage = np.mean(original_label == predicted_label) * 100
    print("DFE Class Validation Accuracy : ", matching_percentage, "%")
    return matching_percentage

def classes(name):
    if name == "Trento":
        classes = {
            0: "annotated apple trees and fruit in hyperspectral images",  # Apples
            1: "identified various manmade structures such as houses commercial buildings and industrial facilities in hyperspectral imagery",  # Buildings
            2: "labeled the earth surface which can include materials like soil grass rocks and natural terrain elements in hyperspectral data",  # Ground
            3: "recognized dense forested areas and enabling analysis of the spectral properties of different tree species and woodland ecosystems in hyperspectral images",  # Woods
            4: "categorized grapevines for monitoring and managing their health and grape quality in hyperspectral images used for wine production",  # Vineyard
            5: "detected paved and unpaved roads for applications such as transportation analysis and infrastructure assessment in hyperspectral images",  # Roads
        }
    if name == "Houston":
        classes = {
            0: "lush green grass thriving and vibrant symbolizing the epitome of natural health in hyperspectral image",  # Healthy grass
            1: "grass exhibiting signs of stress perhaps due to environmental factors or insufficient care in hyperspectral image",  # Stressed grass
            2: "artificial turf resembling natural grass commonly used in various settings for its low maintenance in hyperspectral image",  # Synthetic grass
            3: "tall and majestic trees providing shade and contributing to the ecosystem with their green foliage in hyperspectral image",  # Trees
            4: "the earthy ground a foundation for plant life featuring a mix of minerals and organic matter in hyperspectral image",  # Soil
            5: "a source of life whether it is a serene pond flowing river or any form of liquid sustenance for nature in hyperspectral image",  # Water
            6: "areas designated for housing where communities and families make their homes in hyperspectral image",  # Residential
            7: "spaces designed for business activities from offices to shops fostering economic endeavors in hyperspectral image",  # Commercial
            8: "paved pathways for vehicular transportation connecting destinations and facilitating travel in hyperspectral image",  # Road
            9: "wide and well-traveled roads designed for high-speed long-distance travel between cities and regions in hyperspectral image",  # Highway
            10: "tracks and infrastructure for trains enabling efficient and reliable transportation of goods and passengers in hyperspectral image",  # Railway
            11: "designated areas for parking vehicles providing convenience and organization in hyperspectral image",  # Parking Lot 1
            12: "additional parking space catering to the need for accommodating a larger number of vehicles in hyperspectral image",  # Parking Lot 2
            13: "a sports surface dedicated to tennis featuring marked boundaries and a net for the game in hyperspectral image",  # Tennis Court
            14: "a specially designed track for running often found in sports facilities and schools encouraging physical fitness in hyperspectral image",  # Running Track
        }
    if name == "MUUFL":
        classes = {
            0: "tall and majestic trees providing shade and contributing to the ecosystem with their green foliage in hyperspectral image",  # Trees
            1: "pure grass representing a natural and untarnished ground cover in hyperspectral image in hyperspectral image",  # Grass Pure
            2: "grass on the ground surface depicting the varied textures and patterns of natural landscapes in hyperspectral image",  # Grass Groundsurface
            3: "dirt and sand the granular components of the ground with diverse natural compositions in hyperspectral image",  # Dirt And Sand
            4: "materials used in road construction forming the paved pathways for transportation in hyperspectral image",  # Road Materials
            5: "water a fundamental element in nature whether in the form of lakes rivers or other bodies in hyperspectral image",  # Water
            6: "shadows cast by buildings adding depth and dimension to urban landscapes in hyperspectral image",  # Buildings Shadow
            7: "buildings structures designed for various purposes from residential to commercial in hyperspectral image",  # Buildings
            8: "sidewalks pedestrian pathways along roads and streets facilitating safe walking in hyperspectral image",  # Sidewalk
            9: "yellow curb markings indicating restrictions or specific rules for parking and stopping in hyperspectral image",  # Yellow Curb
            10: "cloth panels possibly used for decorative or functional purposes in outdoor settings in hyperspectral image",  # ClothPanels
        }
    if name == "Augsburg":
        classes = {
            0: "urban residential zones and associated infrastructure captured in hyperspectral images",  # Residential-Area
            1: "areas with sparse vegetation or ground cover depicted in hyperspectral images",  # Low-Plants
            2: "urban commercial districts and buildings showcased in hyperspectral images",  # Commercial-Area
            3: "dense woodland areas and diverse vegetation highlighted in hyperspectral images",  # Forest
            4: "zones with heavy industrial infrastructure and activities depicted in hyperspectral images",  # Industrial-Area
            5: "designated plots for cultivation or gardening purposes featured in hyperspectral images",  # Allotment
            6: "water a fundamental element in nature whether in the form of lakes rivers or other bodies in hyperspectral image",  # Water
        }
    
    return classes

def blip_score(model_path, processor_path, device, val_dataset, name):
    model = torch.load(model_path).to(device)
    processor = torch.load(processor_path)
    num_images = len(val_dataset)
    count = 0

    classes = classes(name)
    classes_swapped = {v: 0 for k, v in classes.items()}
    
    for i, example in tqdm(
        enumerate(val_dataset), desc="Files :", leave=False, total=num_images
    ):
        image = example["image"]
        text = example["text"]
        
        inputs = processor(images=image, return_tensors="pt").to(device)
        pixel_values = inputs.pixel_values

        generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
        generated_caption = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        if text.lower() == generated_caption.lower():
            count += 1
            classes_swapped[generated_caption.lower()]
        
    

        
    matching_percentage = (count / num_images) * 100
    
    print("BLIP Prompt OA accuracy : ", matching_percentage, "%")

    return matching_percentage
