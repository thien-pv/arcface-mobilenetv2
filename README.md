# arcface-mobilenetv2
1. Description

Application of face recognition technique with Arcface model to Smart Camera System.

2. Papers pulished

https://www.researchgate.net/publication/365729132_Application_of_Face_Recognition_Technique_with_ArcFace_model_to_Smart_Camera_System
https://www.researchgate.net/publication/365362063_Smart_home_Management_System_with_Face_Recognition_based_on_ArcFace_model_in_Deep_Convolutional_Neural_Network

3. Dependencies
Install these libraries with anaconda and pip:
- python==3.7
- tensorflow-gpu==2.1.0
- opencv-python==4.1.1.26
- bcolz==1.2.1
- sklearn2
- PyYAML
- matplotlib==3.3.0
- numpy==1.19.0
- pillow==7.2.0
- scikit-learn==0.23.1
- torch==1.5.1
- torchvision==0.6.1
- tqdm==4.47.0

4. Pretrained model
- Download checkpoint for a model from [GoogleDrive/Baidu](https://drive.google.com/drive/folders/1omzvXV_djVIW2A7I09DWMe9JR-9o_MYh) and move it to checkpoint/backbone_ir50_ms1m_epoch120.pth
- Download pretrained model of Asian datasets: https://drive.google.com/file/d/1ABQO2_04zIY0HqF7ElAEH80-xupW96c-/view?usp=share_link 

5. Data
All datasets with faces must support [ImageFolder](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder) format. Look at the prepared examples in data directory.
