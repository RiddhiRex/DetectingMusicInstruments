# DetectingMusicInstrument

Objective:
The project is a Data Sceince project based on Acoustic Sensing to detect what kind of instrument is used in a music piece.
It will identify the instruments played in a music track and classify them as Piano or Guitar or Band or likewise. 

To achieve the goal, we use live recordings of music pieces downloaded from internet. Music is then preprocessed to extract features out of it and then used to train a classifier model. Over 370 features are derived from each music piece which helps in better recognition of music instruments. A music piece can be uploaded to the model and the Classifier will detect the music instrument played in the music piece. 13 different music instruments are predicted with an accuracy of 71%.

Tools used:
This project is developed using Python 2.7. Following tools and libraries were used to develop this project.
1. Pandas
2. Numpy
3. Scikit-Learn (sklearn)
4. Essentia- Open-source library and tools for audio and
music analysis, description and synthesis

Datasets used:
Three different datasets are used to strengthen the mode. They are
     Dataset for Instrument Recognition in Musical Audio
    Signals (IRMAS)
     PhilHarmonia
     University of Iowa Musical Samples (UIOWA MIS)
    
Dataset for Instrument Recognition in Musical Audio Signals (IRMAS):
IRMAS is a dataset specifically intended for doing task like music instrument recognition by training and testing the
classifier to identify the music instruments played in the music piece. IRMAS dataset consists of totally 6705 audio files. The number of files representing each instrument are: Cello(388), Clarinet(505), flute(451), Acoustic Guitar(637), Electric Guitar(760), Organ(682), Piano(721), Saxophone(626), Trumpet(577), Violin(580), Voice(778).

PhilHarmonia:
It consists of 5345 music files in 16 bit stereo wav format sampled at 44.1kHz. This dataset has labels on what instruments are played in each music piece in the dataset. It contains music pieces containing following instruments: cello, clarinet, flute, guitar, saxophone, trumpet, violin.

University of Iowa Musical Samples (UIOWA MIS):
It consists of 816 music files in 16 bit stereo wav format sampled at 44.1kHz. cello, sax, clarinet, flute, guitar, piano.

Feature Extraction using Essentia:
Essentia is an open-source C++ library for audio analysis and audio-based music information retrieval released under the
Affero GPLv3 license which has been developed by the Music Technology Group in University at Pompeu Fabra. Essentia was awarded with the Open-Source Competition of ACM Multimedia in 2013. We used Essentia for musical signal preprocessing and extracting features to help in training classification model.

webUI interface:
I have designed a web interface for easy usage of the users.
We can upload a music file. It will process it and do feature extraction and run it through the model to return what
instruments are played in the music file.
