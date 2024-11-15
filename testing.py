import keras
import numpy as np
import librosa
import soundfile as sf
import tkinter as tk
from tkinter.filedialog import askopenfilename
import shutil
import os
import sys
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import cv2
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

batch_size = 32
img_height = 180
img_width = 180

window = tk.Tk()

window.title("Bird Sound Classification")

window.geometry("500x510")
window.configure(background ="lightgreen")
img=Image.open("o.jpg")
bg=ImageTk.PhotoImage(img)

lbl=tk.Label(window,image=bg)
lbl.place(x=0,y=0)

title = tk.Label(text="Click below to choose audio file for classify the bird....", background = "lightgreen", fg="Brown", font=("", 15))
title.grid()
dirPath = "testpicture"
fileList = os.listdir(dirPath)
dirPath1="image\spectrograms"
fileList1 = os.listdir(dirPath1)
save_path=""
class_names=['bellbird', 'crow', 'eagle', 'hen', 'koyal', 'owl', 'parrot', 'seagull', 'sparrow', 'zorzal']
for fileName in fileList:
        os.remove(dirPath + "/" + fileName)
for file1 in fileList1:
        os.remove(dirPath1 + "/" + file1)
def openphoto():
    
    fileName = askopenfilename(initialdir='data\\', title='Select audio for analysis ',
                           filetypes=[('wav files', '.wav')])

    dst = "testpicture"
    print(fileName)
    print (os.path.split(fileName)[-1])
    if os.path.split(fileName)[-1].split('.') == 'h (1)':
        print('dfdffffffffffffff')
    shutil.copy(fileName, dst)

    def process():
        import os
        import librosa
        import numpy as np
        import matplotlib.pyplot as plt

        # Define the root directory
        root_dir = 'C:\\Users\\moham\\Downloads\\Desktop\\BIRD\\bird_classification\\testpicture'
        path3= 'C:\\Users\\moham\\Downloads\\Desktop\\BIRD\\bird_classification\\image'
        # Define the Mel-spectrogram parameters
        n_fft = 2048
        hop_length = 512
        n_mels = 128

        # Loop through all subdirectories and process each .wav file
        for subdir, dirs, files in os.walk(root_dir):
            for file in files:
                # Check if the file is a .wav file
                if file.endswith('.wav'):
                    # Load the audio file
                    audio_path = os.path.join(subdir, file)
                    y, sr = librosa.load(audio_path)

                    # Compute the Mel-spectrogram
                    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
                    mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)

                    # Plot and save the Mel-spectrogram
                    plt.figure(figsize=(10, 4))
                    plt.imshow(mel_spec_db, origin='lower', aspect='auto', cmap='viridis')
                    plt.colorbar(format='%+2.0f dB')
                    plt.title('Mel-spectrogram')
                    plt.tight_layout()

                    # Save the Mel-spectrogram as an image file
                    save_dir = os.path.join(path3, 'spectrograms')
                    global save_path
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    save_path = os.path.join(save_dir, os.path.splitext(file)[0] + '.png')
                    
                    plt.savefig(save_path)
                    plt.close()
        #os.rename(save_path,"test.png")
    def audio():
        import pickle
        #global save_path
        model=pickle.load(open("C:\\Users\\moham\\Downloads\\Desktop\\BIRD\\bird_classification\\trained.pkl","rb"))
        input_path="C:\\Users\\moham\\Downloads\\Desktop\\BIRD\\bird_classification\\image\\spectrograms\\"
        import os
        from os import listdir
 
        #folder_dir = "C:/Users/RIJUSHREE/Desktop/Gfg images"
        for images in os.listdir(input_path):
            if (images.endswith(".png")):
                print(images)

        save_path=input_path+str(images)
        print("fff{}".format(save_path))
        img = tf.keras.utils.load_img(
            save_path, target_size=(img_height, img_width)
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) #

        out=model.predict([img_array])
        #out1=model.predict_proba([img_array])
        score = tf.nn.softmax(out[0])
        #print(out[0])
        #print(out1[0]*100)
        #print(score)
        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
        )
        if class_names[np.argmax(score)]=='bellbird':
                im=cv2.imread("bellbird.jpg")
                cv2.imshow("out",im)
        if class_names[np.argmax(score)]=='crow':
                im=cv2.imread("crow.jpg")
                cv2.imshow("out",im)
        if class_names[np.argmax(score)]=='eagle':
                im=cv2.imread("eagle.jpg")
                cv2.imshow("out",im)
        if class_names[np.argmax(score)]=='hen':
                im=cv2.imread("hen.jpg")
                cv2.imshow("out",im)
        if class_names[np.argmax(score)]=='koyal':
                im=cv2.imread("koyal.jpg")
                cv2.imshow("out",im)
        if class_names[np.argmax(score)]=='owl':
                im=cv2.imread("owl.jpg")
                cv2.imshow("out",im)
        if class_names[np.argmax(score)]=='parrot':
                im=cv2.imread("parrot.jpg")
                cv2.imshow("out",im)
        if class_names[np.argmax(score)]=='sparrow':
                im=cv2.imread("sparrow.jpg")
                cv2.imshow("out",im)
        if class_names[np.argmax(score)]=='zorzal':
                im=cv2.imread("zorzal.jpg")
                cv2.imshow("out",im)
        if class_names[np.argmax(score)]=='seagull':
                im=cv2.imread("seagull.jpg")
                cv2.imshow("out",im)
        disease = tk.Label(text='THE PREDICTED BIRD IS :' + class_names[np.argmax(score)], background="darkcyan",
                               fg="Red", font=("", 15))
        disease.grid(column=0, row=4, padx=20, pady=20)
        button = tk.Button(text="Exit", command=exit)
        button.grid(column=0, row=9, padx=20, pady=20)
    button2 = tk.Button(text="predict", command = audio)
    button2.grid(column=0, row=2, padx=10, pady = 10)
    button3 = tk.Button(text="process", command = process)
    button3.grid(column=0, row=0, padx=10, pady = 10)


button1 = tk.Button(text="Choose file", command = openphoto)
button1.grid(column=0, row=1, padx=10, pady = 10)



window.mainloop()
