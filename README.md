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
6. Data preprocessing and training

To prepare data with cropped and aligned faces from your original images, run:

python face_alignment.py --tags Face_main/data/student --crop_size 112

To use t-SNE for dimensionality reduction and 2D visualization of face embeddings, run:

python tsne.py --tags student

File named embeddings.pickle in ouput when traning:
python Face_main/train_model.py --embeddings Face_main/output/embeddings.pickle --recognizer ./Face_main/output/recognizer.pickle --le ./Face_main/output/le.pickle

Runing arcface model with python ./Face_main/run.py

In file run.py, you can replace pretrained models with SSD, RetinaFace to detect faces or MTCNN. But, SSD and Retinaface are more good for results about FPS. Also, in our code, you can completely fix it if you get any errors.  
Above, you can run and train it on your window laptop
We have deployed code in 2 papers on Jetson Nano 4GB with 8MP Sony IMX219 77 degree CS Camera. Results, you can follow and read it in our papers (2. papers)

Thank you so much!

