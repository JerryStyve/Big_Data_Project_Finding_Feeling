import librosa
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
MODEL_PATH="model.h5"
NUM_SAMPLES_TO_CONSIDER=44100 #1sec
class finding_feelings:
    model=None
    _mappings= [
        "Angry",
        "Calm",
        "Disgust",
        "Fearful",
        "Happy",
        "Neutral",
        "Sad",
        "Surprise"
    ]
    _instance=None

    def predict(self,file_path):
        #extract the MFCCs
        MFCCs=self.preprocess(file_path)


        #convert 2D MFCCs array to 4D array(samples,segments,coeffs,channel)
        MFCCs=MFCCs[np.newaxis,...,np.newaxis]



        #make prediction
        predictions=self.model.predict(MFCCs)#
        predicted_index=np.argmax(predictions)
        predicted_feeling=self._mappings[predicted_index]

        return predicted_feeling

    def preprocess(self,file_path,n_mfcc=40,n_fft=2048,hop_length=512):
        # load audio
        signal,sr=librosa.load(file_path)
        # ensure consistency in the audio file length
        if len(signal)>NUM_SAMPLES_TO_CONSIDER:
            signal=signal[:NUM_SAMPLES_TO_CONSIDER]
            # extract a MFCCs
        MFCCs=librosa.feature.mfcc(signal, sr, n_mfcc=n_mfcc,n_fft=n_fft,hop_length=hop_length)
        return MFCCs.T


def finding_feelings_service():

    #ensure that e only hae one instance of ffs
    if finding_feelings._instance is None:
        finding_feelings._instance=finding_feelings()
        finding_feelings.model=tf.keras.models.load_model(MODEL_PATH)
        return finding_feelings._instance

if __name__=="__main__":
    ffs=finding_feelings_service()
    feeling=ffs.predict("Dataset/Happy/OAF_room_happy.wav")
    feeling2=ffs.predict("Dataset/Sad/OAF_bite_sad.wav")
    print(f"predicted feelings:{feeling},{feeling2}")#print prediction