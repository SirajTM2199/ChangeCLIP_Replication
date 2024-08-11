#model, preprocess = clip.load("ViT-B/32", device=device)
from CLIP import clip 
import torch
import torch.nn as nn
def Prompt_Maker(Initial_Image,Final_Image,model,preprocess,not_done=False):
    Outputs = [["remote sensing image foreground objects"],
               ["remote sensing image background objects"],
               ["remote sensing image foreground objects"],
               ["remote sensing image background objects"]]
    TEXT= clip.tokenize(Classes).to(device)
    images = [Initial_Image,Final_Image]
   
    for i in range(len(images)):
        results = []
        if not_done:
            IMAGE = preprocess(Image.open(images[i])).unsqueeze(0).to(device)
        else:
            IMAGE = images[i]
        with torch.no_grad():
            image_features,pos = model.encode_image(IMAGE,return_pos=True)
            text_features = model.encode_text(TEXT)
            
            logits_per_image, logits_per_text = model(IMAGE,TEXT)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        #print(type(Outputs))   
        Outputs[2*i][0] += (",")
        for index in np.argsort(probs)[0][::-1][:9]:
            #results.append()
            Outputs[2*i][0] += Classes[index]
            Outputs[2*i][0] += ","
        Outputs[2*i][0] = Outputs[2*i][0][:-1]
        #Outputs[2*i][0] += ("}")
    return Outputs