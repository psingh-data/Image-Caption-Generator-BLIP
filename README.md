Image Caption Generator using BLIP

A deep-learning project that generates natural language captions from images using the BLIP (Bootstrapping Language-Image Pre-training) model.

This repository includes:
- A complete Jupyter notebook for training/evaluating BLIP  
- A saved BLIP model package with configs & metadata  
- Quick reference usage guide  
- MIT License  



Features

- State-of-the-art BLIP architecture
- High-quality captions trained/evaluated on the Flickr8k dataset
- GPU + CPU compatible
- No training required if using the provided model
- Supports batch image captioning
- Production-ready model files included



📁 Project Structure
Image-Caption-Generator-BLIP/
│
├── notebooks/
│ └── Image_Caption_Generator_BLIP.ipynb
│
├── model/
│ └── Image_caption_generator_blipmodel/
│ ├── pytorch_model.bin
│ ├── config.json
│ ├── tokenizer_config.json
│ ├── preprocessor_config.json
│ ├── vocab.txt
│ ├── model_info.json
│ ├── README.md
│ └── QUICK_REFERENCE.txt
│
├── examples/
│ ├── sample.png
│ ├── output_caption.txt
│
├── requirements.txt
├── .gitignore
└── LICENSE


---

Installation
pip install -r requirements.txt

---

##  How to Use

# Load the Model
python
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

model_path = "model/Image_caption_generator_blipmodel"

processor = BlipProcessor.from_pretrained(model_path)
model = BlipForConditionalGeneration.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Generate a Caption
image = Image.open("your_image.jpg").convert("RGB")
inputs = processor(image, return_tensors="pt").to(device)

output = model.generate(**inputs, max_length=50, num_beams=5)
caption = processor.decode(output[0], skip_special_tokens=True)

print("Caption:", caption)


# Model Performance (Flickr8k)
Metric	Score
BLEU-1	0.55+
BLEU-2	0.40+
BLEU-3	0.30+
BLEU-4	0.25+

(Exact values inside model_info.json)

License

This project is open-source under the MIT License (see LICENSE file).

👤 Author

Prabhjot Singh
2025


---

# LICENSE (MIT License)



MIT License

Copyright (c) 2025 Prabhjot Singh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


# requirements.txt (Minimal + Works Everywhere)
torch
torchvision
transformers
pillow
numpy
pandas
tqdm
scikit-learn


---
# .gitignore (Best for ML Projects)


Python

pycache/
*.pyc

Model Weights

*.bin
*.pt

Jupyter checkpoints

.ipynb_checkpoints/

OS junk

.DS_Store
Thumbs.db

Environments

.env
.venv/

Logs

*.log



