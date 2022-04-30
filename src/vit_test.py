import json
from PIL import Image

import torch
from torchvision import transforms

from pytorch_pretrained_vit import ViT
import matplotlib.pyplot as plt
import cv2
import pandas as pd

cuda0 = torch.device('cuda:0')

model_name = 'B_16_imagenet1k'
model = ViT(model_name, pretrained=True)

# Load class names
labels_map = json.load(open('./src/labels_map.txt'))
labels_map = [labels_map[str(i)] for i in range(1000)]

# img = Image.open('./src/bovik.png')
# img
def read_photo(path):

    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)

    # Preprocess image
    tfms = transforms.Compose([transforms.Resize(model.image_size), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),])
    img = tfms(img).unsqueeze(0)
    # transform = transforms.Compose([transforms.ToTensor()])
    # img = transform(img)
    # img = torch.reshape(img, (1, model.image_size)).to(cuda0)

    # Classify
    model.eval()
    with torch.no_grad():
        outputs = model(img).squeeze(0)
    print('-----')
    idx_list, prob_list = [], []
    for idx in torch.topk(outputs, k=100).indices.tolist():
        prob = torch.softmax(outputs, -1)[idx].item()
        idx_list.append(idx)
        prob_list.append(prob)
        #print('[{idx}] {label:<75} ({p:.2f}%)'.format(idx=idx, label=labels_map[idx], p=prob*100))

    return dict(zip(idx_list, prob_list))

og_dict = read_photo('./src/comp/original_image.png')
gen_dict = read_photo('./src/comp/image_29751.png')

match_keys = list(set(og_dict.keys()).intersection(set(gen_dict.keys())))
og_match = {key: og_dict[key] for key in match_keys}
gen_match = {key: gen_dict[key] for key in match_keys}

df = pd.DataFrame([og_match, gen_match]).T
labels = [labels_map[key] for key in df.index.to_list()]
df.insert(0, "Label", labels)
df.columns = ['Label', 'Original Probability','Generated Probability']
df['Difference in Probability'] = abs(df.iloc[:,1] - df.iloc[:,2])
df.sort_values(by="Original Probability", ascending=False)