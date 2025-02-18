import torch
import torch.nn as nn
from PIL import Image
import clip
import openai
import ImageReward as RM
from mllm_template import template
import base64
import io
import json
import re

class CLIPScoreReward:
    def __init__(self, device="cuda"):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-L/14", device=device)
        
    def get_reward(self, images, prompts):
        """Calculate CLIP Score
        Args:
            images: List of PIL Images or single image
            prompts: List of text prompts or single prompt
        Returns:
            reward scores
        """
        if isinstance(images, Image.Image):
            images = [images]
        if isinstance(prompts, str):
            prompts = [prompts]
            
        image_inputs = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        text_inputs = clip.tokenize(prompts).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_inputs)
            text_features = self.model.encode_text(text_inputs)
            
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            similarity = (100.0 * image_features @ text_features.T)
        return similarity.diagonal().cpu()

class ImageRewardModel:
    def __init__(self, device="cuda"):
        self.device = device
        self.model = RM.load("ImageReward-v1.0")
        
    def get_reward(self, images, prompts):
        """Calculate ImageReward scores
        Args:
            images: List of PIL Images or single image
            prompts: List of text prompts or single prompt
        Returns:
            reward scores
        """
        if isinstance(images, Image.Image):
            images = [images]
        if isinstance(prompts, str):
            prompts = [prompts]
            
        scores = []
        for img, prompt in zip(images, prompts):
            score = self.model.score(prompt, img)
            scores.append(score)
            
        return torch.tensor(scores)

class CompositionalReward:
    def __init__(self, device="cuda"):
        """Compositional reward that combines multiple metrics"""
        self.device = device
        self.model = RM.load("path_to_your_trained_model")
        
    def get_reward(self, images, prompts):
        """Calculate combined reward
        Args:
            images: List of PIL Images or single image
            prompts: List of text prompts or single prompt
            weights: Dictionary of weights for different rewards
        Returns:
            combined reward scores
        """
        if isinstance(images, Image.Image):
            images = [images]
        if isinstance(prompts, str):
            prompts = [prompts]
            
        scores = []
        for img, prompt in zip(images, prompts):
            score = self.model.score(prompt, img)
            scores.append(score)
            
        return torch.tensor(scores)

class MLLMGrader:
    def __init__(self, base_url=None, api_key=None):
        """GPT-4V based grader
        Args:
            base_url: OpenAI API base URL (optional)
            api_key: OpenAI API key
        """
        # Configure OpenAI settings
        if base_url:
            openai.api_base = base_url
        if api_key:
            openai.api_key = api_key
        
        # Initialize new OpenAI client
        from openai import OpenAI
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url if base_url else "https://api.openai.com/v1"
        )
    
    def get_reward(self, images, prompts):
        """Use GPT-4V to grade image quality
        Args:
            images: List of PIL Images or single image
            prompts: List of text prompts or single prompt
        Returns:
            reward scores
        """
        if isinstance(images, Image.Image):
            images = [images]
        if isinstance(prompts, str):
            prompts = [prompts]
            
        scores = []
        for img, prompt in zip(images, prompts):
            # Convert PIL Image to base64
            buffered = io.BytesIO() 
            img.save(buffered, format="PNG")
            base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": template.format(prompt=prompt)
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
            
            # Use new client API call method
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=1000,
            )
            
            try:
                # Get the raw content first
                content = response.choices[0].message.content.strip()   
                numbers = re.findall(r"[-+]?\d*\.\d+|\d+", content)
                score = float(numbers[-1]) if numbers else 7.0
                scores.append(score)
            except Exception as e:
                print(f"Error parsing response: {e}")  # Debug print
                scores.append(7.0)  # Default score if parsing fails
                
        return torch.tensor(scores)



# reward_model_funcs = {
#     "clip": CLIPScoreReward,
#     "image_reward": ImageRewardModel,
#     "compositional": CompositionalReward,
#     "mllm": MLLMGrader
# }

# reward_model = reward_model_funcs["mllm"](base_url="https://api2.aigcbest.top/v1", api_key="sk-Y8khhuxVHuPDYn8A2461E99561104a71Bc4a917a664f7aB5")
# img = Image.open("/mnt/bn/automl-aigc/tianye/ByteGEN_AIGC/acceleration/best_image_2.png")
# width = img.size[0]
# left_half = img.crop((0, 0, width//2, img.size[1]))
# right_half = img.crop((width//2, 0, width, img.size[1]))
# images = [left_half, right_half]
# prompts = ["a beautiful woman"] * 2
# scores = reward_model.get_reward(images, prompts) 
# print(scores)
