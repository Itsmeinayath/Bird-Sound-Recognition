import tkinter as ttk
from tkinter import *
import cv2
from PIL import Image, ImageTk
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
from tensorflow.keras import layers,models
from tensorflow.keras.models import Sequential
batch_size = 32
img_height = 180
img_width = 180

class App:
    def __init__(self, window, window_title, video_source):
        self.window = window
        self.window.title(window_title)
        
        # Open the video source
        self.cap = cv2.VideoCapture(video_source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1500)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 750)
        
        # Create a canvas that can fit the video source
        self.canvas = ttk.Canvas(window, width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), 
                                height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()
        
        # Use PIL (Pillow) to convert the OpenCV image to a Tkinter image
        self.photo = None
        self.update()
        
        import warnings
        warnings.filterwarnings('ignore')
        window.geometry("1600x850")

        
        title = tk.Label(text="Click below to choose audio file for \n classify the bird....", background = "#9edfe8", fg="black", font=("times", 24,"underline italic"))
        title.place(x=200,y=30)
        head = tk.Label(text=" BIRD \n\n S \n O \n U \n N \n D \n\n R \n E \n C \n O \n G \n N \n I \n T \n I \n O \n N",width=7, background = "#9edfe8",
                        fg="black", font=("elephant", 22," bold"))
        head.place(x=0,y=0)
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
        def play_det(bird_name):
            import pygame
            from gtts import gTTS

            tts = gTTS(bird_name)
            tts.save("hello.wav")
            pygame.init()

            sound = pygame.mixer.Sound("hello.wav")
            sound.play()

            while pygame.mixer.get_busy():
                pygame.time.wait(100)

            pygame.quit()
            window.mainloop()
        def play_aud(audio):
            import pygame

            pygame.init()

            sound = pygame.mixer.Sound(audio)
            sound.play()

            while pygame.mixer.get_busy():
                pygame.time.wait(100)

            pygame.quit()
            window.mainloop()
        def secopenphoto():
            lblo.destroy()
            lbls.destroy()
            disease.destroy()
            openphoto()
            buttonr.destroy()
        def openphoto():
            button1.destroy()
            title.configure(text="")
            fileName = askopenfilename(initialdir='data\\', title='Select audio for analysis ',
                                   filetypes=[('wav files', '.wav')])

            dst = "testpicture"
            print(fileName)
            print (os.path.split(fileName)[-1])
            if os.path.split(fileName)[-1].split('.') == 'h (1)':
                print('dfdffffffffffffff')
            shutil.copy(fileName, dst)
            
            def audio():
                button4.destroy()
               
                #global save_path
                from tensorflow import keras
                model = keras.models.load_model('model.h5')
                #model=pickle.load(open("C:\\Users\\ramesh\\Desktop\\BIRD CLASSIFICATION\\trained.pkl","rb"))
                #model=pickle.loads(("C:\\Users\\ramesh\\Desktop\\BIRD CLASSIFICATION\\trained.pkl"))

                input_path="image\\spectrograms\\"
                import os
                from os import listdir
         
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
                
                bird_n=class_names[np.argmax(score)]
                out="THE PREDICTED BIRD IS :\n " + str(bird_n)
                im=Image.open(str(bird_n)+".jpg")
                im=im.resize((300,300))
                bgg=ImageTk.PhotoImage(im)
                global lblo,disease,buttonr
                lblo=tk.Button(window,image=bgg,command=lambda:play_det(out))
                lblo.place(x=800,y=180)
                lblo.image=bgg
                
                disease = tk.Label(text='THE PREDICTED BIRD IS :\n ' + bird_n, background = "#9edfe8", fg="black", font=("times", 24,"underline italic"))
                disease.place(x=800,y=580)
                button = tk.Button(text="Exit", command=exit,background = "#9edfe8", fg="black", font=("times", 24,"underline italic"))
                button.place(x=300,y=580)
                buttonr = tk.Button(text="Recheck", command=secopenphoto,background = "#9edfe8", fg="black", font=("times", 24,"underline italic"))
                buttonr.place(x=300,y=580)

            def process():
                button3.destroy()
                import os
                import librosa
                import numpy as np
                import matplotlib.pyplot as plt

                # Define the root directory
                root_dir = 'testpicture'
                path3= 'image'
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
                            print(save_path)
                            plt.savefig(save_path)
                            plt.close()
                            imm=Image.open(str(save_path))
                            imm=imm.resize((600,300))
                            bggg=ImageTk.PhotoImage(imm)
                            global lbls
                            lbls=tk.Button(window,image=bggg,command=lambda:play_aud(audio_path))
                            lbls.place(x=150,y=180)
                            lbls.image=bggg
                            title.configure(text="click 'Analyse Mfcc' Button to classify the sound of bird using CNN")
                            global button4
                            button4 = tk.Button(text="Analyse Mfcc", command = audio,background = "#9edfe8", fg="black", font=("times", 24,"underline italic"))
                            button4.place(x=300,y=500)
                            
                #os.rename(save_path,"test.png")
            title.configure(text="click 'Convert to Mfcc' Button to convert the sound to mfcc using librosa")
            button3 = tk.Button(text="Convert to Mfcc", command = process,background = "#9edfe8", fg="black", font=("times", 24,"underline italic"))
            button3.place(x=300,y=200)
            


        button1 = tk.Button(text="Choose file", command = openphoto,background = "#9edfe8", fg="black", font=("times", 24,"underline italic"))
        button1.place(x=300,y=150)


 
        window.mainloop()

        # Start the video playback loop
        self.window.mainloop()
    
    def update(self):
        # Get a frame from the video source
        ret, frame = self.cap.read()
        frame = cv2.resize(frame, (1600,850), interpolation = cv2.INTER_AREA)
        if ret:
            self.photo = ImageTk.PhotoImage(image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image = self.photo, anchor = ttk.NW)
        
        # Repeat after 15 milliseconds
        self.window.after(15, self.update)

# Create a window and pass it to the Application object
App(ttk.Tk(), "Tkinter Video Looping Background", "video.mp4")

