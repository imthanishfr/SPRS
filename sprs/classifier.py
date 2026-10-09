"""
Zero-Shot Image Classification via CLIP (sprs/classifier.py)

Classifies cropped object images against target categories:
- Electronics
- Fragile (Glass/Ceramics)
- Toys
- Books/Stationery
- Clothing/Textiles
- Other/Miscellaneous

Uses HuggingFace CLIP (openai/clip-vit-base-patch32) with zero-shot prompt matching.
"""

import os
import cv2
import numpy as np
from PIL import Image
from typing import Dict, Tuple, Any, List, Optional, Union

# Candidate target categories matching rules matrix
CATEGORIES_LIST = [
    "Electronics",
    "Mobile Devices/Earbuds",
    "Fragile (Glass/Ceramics)",
    "Toys/Plastics",
    "Books/Stationery",
    "Clothing/Textiles",
    "Heavy Tools/Hardware",
    "Other/Miscellaneous"
]

# Engineered multi-keyword zero-shot text prompts for high-accuracy CLIP matching
CATEGORY_PROMPTS = {
    "Electronics": "a photo of an electronic device, tech product, charger, power bank, remote control, circuit board, or digital hardware",
    "Mobile Devices/Earbuds": "a photo of a mobile phone, smartphone, wireless earbuds case, bluetooth headphone case, smart watch, or compact electronic accessory",
    "Fragile (Glass/Ceramics)": "a photo of fragile glassware, ceramic mug, porcelain dish, glass bottle, wine glass, crystal, or vase",
    "Toys/Plastics": "a photo of a plastic toy, action figure, plush doll, plaything, or molded plastic item",
    "Books/Stationery": "a photo of a book, hardcover book, notebook, printed paper, stationery box, or magazine",
    "Clothing/Textiles": "a photo of a shirt, pants, jacket, folded fabric garment, apparel, towel, or cloth bag",
    "Heavy Tools/Hardware": "a photo of a metal tool, wrench, hammer, power tool, hardware component, or machinery part",
    "Other/Miscellaneous": "a photo of a general household object or item"
}


class ClipCategoryClassifier:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None
        self.initialized = False
        self._load_model()

    def _load_model(self):
        """Lazy loader for CLIP model and processor with graceful fallback."""
        try:
            import torch
            from transformers import CLIPProcessor, CLIPModel

            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            print(f"[SPRS Classifier] Loading CLIP model '{self.model_name}' on {self.device}...")
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            try:
                self.model = CLIPModel.from_pretrained(self.model_name, use_safetensors=True).to(self.device)
            except Exception:
                self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            self.initialized = True
            print("[SPRS Classifier] CLIP model successfully loaded on GPU/CPU.")
        except Exception as e:
            print(f"[SPRS Classifier Warning] Could not load CLIP model: {e}")
            print("[SPRS Classifier Warning] Falling back to rule/heuristic classification mode.")
            self.initialized = False

    def classify(
        self,
        image_input: Union[Image.Image, np.ndarray],
        heuristic_label: Optional[str] = None
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Classify input image into target categories using CLIP + YOLO score fusion.
        """
        if isinstance(image_input, np.ndarray):
            if image_input.size == 0:
                return "Other/Miscellaneous", 0.50, {cat: 0.12 for cat in CATEGORIES_LIST}
            
            # Preprocess crop: square pad image to preserve aspect ratio
            h, w = image_input.shape[:2]
            max_side = max(h, w)
            padded = np.zeros((max_side, max_side, 3), dtype=np.uint8)
            yo = (max_side - h) // 2
            xo = (max_side - w) // 2
            padded[yo:yo+h, xo:xo+w] = image_input

            rgb_arr = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_arr)
        elif isinstance(image_input, Image.Image):
            pil_img = image_input
        else:
            return "Other/Miscellaneous", 0.50, {cat: 0.12 for cat in CATEGORIES_LIST}

        if not self.initialized or self.model is None or self.processor is None:
            return self._heuristic_fallback(heuristic_label)

        try:
            import torch

            # 1. DIRECT COCO YOLO OVERRIDE (High Precision Label Mapping)
            if heuristic_label:
                hint = heuristic_label.lower().strip()
                if any(w in hint for w in ["phone", "cell phone", "remote", "mouse", "keyboard", "earbuds", "headphone"]):
                    scores = {cat: 0.02 for cat in CATEGORIES_LIST}
                    scores["Mobile Devices/Earbuds"] = 0.96
                    scores["Electronics"] = 0.92
                    return "Mobile Devices/Earbuds", 0.96, scores
                elif any(w in hint for w in ["laptop", "tv", "monitor", "clock", "charger", "power bank"]):
                    scores = {cat: 0.02 for cat in CATEGORIES_LIST}
                    scores["Electronics"] = 0.95
                    return "Electronics", 0.95, scores
                elif any(w in hint for w in ["bottle", "wine glass", "vase"]):
                    scores = {cat: 0.02 for cat in CATEGORIES_LIST}
                    scores["Glassware/Bottles"] = 0.94
                    scores["Fragile (Glass/Ceramics)"] = 0.88
                    return "Glassware/Bottles", 0.94, scores
                elif any(w in hint for w in ["cup", "bowl", "ceramic", "mug"]):
                    scores = {cat: 0.02 for cat in CATEGORIES_LIST}
                    scores["Fragile (Glass/Ceramics)"] = 0.92
                    return "Fragile (Glass/Ceramics)", 0.92, scores
                elif any(w in hint for w in ["book", "paper", "notebook", "scissors"]):
                    scores = {cat: 0.02 for cat in CATEGORIES_LIST}
                    scores["Books/Stationery"] = 0.94
                    return "Books/Stationery", 0.94, scores
                elif any(w in hint for w in ["backpack", "handbag", "suitcase", "tie", "t-shirt", "garment"]):
                    scores = {cat: 0.02 for cat in CATEGORIES_LIST}
                    scores["Clothing/Textiles"] = 0.92
                    return "Clothing/Textiles", 0.92, scores

            prompts = [CATEGORY_PROMPTS[cat] for cat in CATEGORIES_LIST]
            
            inputs = self.processor(
                text=prompts,
                images=pil_img,
                return_tensors="pt",
                padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = (logits_per_image / 0.8).softmax(dim=1).cpu().numpy()[0]

            scores_dict = {
                cat: round(float(prob), 4)
                for cat, prob in zip(CATEGORIES_LIST, probs)
            }

            top_idx = int(np.argmax(probs))
            top_category = CATEGORIES_LIST[top_idx]
            confidence = round(float(probs[top_idx]), 4)

            return top_category, confidence, scores_dict

        except Exception as e:
            print(f"[SPRS Classifier Error] Error during CLIP inference: {e}")
            return self._heuristic_fallback(heuristic_label)

    def _heuristic_fallback(self, label_hint: Optional[str] = None) -> Tuple[str, float, Dict[str, float]]:
        """Fallback mapper when CLIP is unavailable or loading."""
        scores = {cat: 0.10 for cat in CATEGORIES_LIST}
        
        if label_hint:
            hint = label_hint.lower()
            if any(w in hint for w in ["laptop", "phone", "cell phone", "keyboard", "mouse", "tv", "monitor", "remote", "clock", "electronic"]):
                scores["Electronics"] = 0.85
                return "Electronics", 0.85, scores
            elif any(w in hint for w in ["bottle", "cup", "wine glass", "bowl", "glass", "vase", "ceramic"]):
                scores["Fragile (Glass/Ceramics)"] = 0.85
                return "Fragile (Glass/Ceramics)", 0.85, scores
            elif any(w in hint for w in ["toy", "teddy bear", "sports ball", "frisbee", "kite"]):
                scores["Toys"] = 0.85
                return "Toys", 0.85, scores
            elif any(w in hint for w in ["book", "paper", "scissors"]):
                scores["Books/Stationery"] = 0.85
                return "Books/Stationery", 0.85, scores
            elif any(w in hint for w in ["backpack", "handbag", "suitcase", "tie", "couch", "bed"]):
                scores["Clothing/Textiles"] = 0.85
                return "Clothing/Textiles", 0.85, scores

        scores["Other/Miscellaneous"] = 0.60
        return "Other/Miscellaneous", 0.60, scores
