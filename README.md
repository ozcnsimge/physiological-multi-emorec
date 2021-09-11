# The Role of Physiological Signals in Multimodal Emotion Recognition Solutions in the Era of Autonomous Driving

Emotions are complex reaction patterns involving observational, behavioral and physiological elements.
Affect and emotion are fundamental components of a human being and they play an important role in everyday lives, for example in computer-related interaction. With the rapid development of autonomous systems and artificial intelligence, more elaborate human-machine interfaces are becoming prominent.
In particular, it is crucial for an autonomous system to understand and react to human emotions.
One such application is an intelligent assistant which can react to the driver’s negative emotional states and adjust the driving behavior according to the emotional state of the driver.
The goal of this thesis is to evaluate and study the overall role and effects of the physiological signals in an in-vehicle multimodal emotion recognition system. 
To this end, we implemented three end-to-end deep learning architectures specialized for time-series classification and compared the experimental results with the raw physiological signals. The best performing architecture, Spectro-Temporal Residual Network, reached the accuracy of 66\% on classifying the anger, sadness, contempt, and fear emotional states. Later, this network was fused into a multimodal emotion recognition model. By fusing physiological modality with behavioral and facial modalities, we were able to improve the accuracy by 23\%. We implemented a preprocessing pipeline for the utilized database "_A multimodal dataset for various forms of distracted driving_" which can be reused in future research in this field.

# Guide 

- Clone this repository and change to that directory.
```bash
git clone https://gitlab.lrz.de/Thesis/physiological-multi-emorec.git
cd physiological-multi-emorec
```
- Download the data which includes synced data and the preprocessed .avi files from https://syncandshare.lrz.de/dl/fiVauGyqxBaaZUc9ndC7gKgG/study_data.zip and unzip it.
```bash
wget https://syncandshare.lrz.de/dl/fiVauGyqxBaaZUc9ndC7gKgG/study_data.zip
unzip study_data.zip
```
- Build and run the Dockerfile
```bash
./build_image.sh
./run_docker.sh
```
- All the training and evaluting scripts can now be run in the docker container:
- Before training Stresnet, Resnet and FCN, preprocess the data by running preprocess.py
```bash
python preprocess.py
```
- To train the Stresnet, Resnet and FCN, run [tuning.sh](tuning.sh)
```bash
./tuning.sh
```
- To train or evaluate the fused model:
```bash
cd fusion

# to evaluate
python fusion_negatives.py false

#to train
python fusion_negatives.py true
```
- To train or to evaluate the the individual modalities:
```bash
cd fusion
python evaluate_physio_model.py stresnet
python evaluate_physio_model.py fcn
python evaluate_physio_model.py resnet

# to evaluate
python train_behavioral_cnn.py false

#to train
python train_behavioral_cnn.py true

# to evaluate
python train_facial_cnn.py false

#to train
python train_facial_cnn.py true
```

### Final note
Training and evaluation results may vary mostly due to:
- Machine learning algorithms train different models if the training dataset is changed.
- Stochastic machine learning algorithms use randomness during learning, ensuring a different model is trained each run.
- Differences in the development environment, such as software versions and CPU type, can cause rounding error differences in predictions and model evaluations.

# References

- Dziezyc, Maciej & Gjoreski, Martin & Kazienko, Przemysław & Saganowski, Stanisław & Gams, Matjaz. (2020). Can We Ditch Feature Engineering? End-to-End Deep Learning for Affect Recognition from Physiological Sensor Data. Sensors. 20. 6535. 10.3390/s20226535. 
- Tzirakis, Panagiotis & Trigeorgis, George & Nicolaou, Mihalis & Schuller, Björn & Zafeiriou, Stefanos. (2017). End-to-End Multimodal Emotion Recognition Using Deep Neural Networks. IEEE Journal of Selected Topics in Signal Processing. PP. 10.1109/JSTSP.2017.2764438. 
- Arriaga, Octavio & Valdenegro, Matias & Plöger, Paul. (2017). Real-time Convolutional Neural Networks for Emotion and Gender Classification.
- S. Taamneh, P. Tsiamyrtzis, M. Dcosta, P. Buddharaju, A. Khatri, M. Manser, T.Ferris, R. Wunderlich, and I. Pavlidis. “A multimodal dataset for various formsof distracted driving.” In:Scientific Data4.1 (2017), p. 170110.

