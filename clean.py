import os
import torch

model_folder = torch.hub.get_dir()
hf_folder = os.path.join(model_folder, 'hf')

if os.path.exists(hf_folder):
    for file in os.listdir(hf_folder):
        file_path = os.path.join(hf_folder, file)
        try:
            if os.path.isfile(file_path):
                if file_path.endswith('.bin') or file_path.endswith('.pt'):
                    os.remove(file_path)
        except Exception as e:
            print(e)
else:
    print("Hugging Face models folder does not exist.")
