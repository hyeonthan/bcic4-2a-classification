# 1. Title
## Classification of Motor Imagery-based EEG Signals

# 2. Introduction
## 2-1. Objective
* Proposal for improving classification performance of decoding EEG signals in the subject-independent environment
## 2-2. Motivation
In subject-independent envrionment, the primary goal of improving classification performance is to develop a brain signal classification model that can be applied to a diverse range of users. This effort entails creating a universal model, rather than one trained for specific users, which can be utilized by a wide range of individuals. This allows BCI systems to become more accessible and versatile, finding applications in various fields such as assistive devices for people with disabilities, computer control, medical applications, and more.
# 3. Data description
This dataset contains EEG data collected from 9 subjects. The cue-based BCI paradigm involved four distinct motor imagery tasks: imagining movement of the left hand (class 0), right hand (class 1), both feet (class 2), and tongue (class 3). For each subject, two sessions were recorded on different days. Each session consisted of 6 runs with short breaks in between. One run included 48 trials, with 12 trials for each of the four possible classes, resulting in a total of 288 trials per session.

Twenty-two Ag/AgCl electrodes (with inter-electrode distances of 3.5cm) were used to record the EEG. All signals were recorded monopolarly with the left mastoid serving as reference and the right mastoid as ground. The signals were sampled with 250 Hz and bandpass-filtered between 0.5 Hz and 100 Hz. The sensitivity of the amplifier was set to 100 μV. An additional 50 Hz notch filter was enabled to suppress line noise.

The dataset was preprocessed and saved in numpy format. For training, the data of 8 out of 9 subjects were trained, and the training and evaluation data of the remaining 1 subject were categorized as validation and test, respectively. The *_X.npy file is the data to be used for training and the *_Y.npy file is the label.
