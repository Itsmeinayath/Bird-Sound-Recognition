# Bird Sound Recognition Model Based on Deep Learning Methodology

## Introduction

This project aims to recognize bird sounds using a deep learning methodology. It provides a graphical user interface (GUI) for users to upload audio files, analyze them, and identify the corresponding bird species.

---

## Prerequisites

1. **Python Version**: Install Python 3.9.0 and ensure it is added to your system path.
2. **Code Editor**: Download and install Visual Studio Code (VS Code) for an optimal development experience.

---

## Required Python Libraries

The following libraries are required to run the project. You can install them using the `pip install` command:

```bash
bash
Copy code
pip install <library_name>

```

Example:

```bash
bash
Copy code
pip install pygame

```

### Libraries Used:

- absl-py==1.4.0
- aiohttp==3.8.4
- aiosignal==1.3.1
- audioread==3.0.0
- Flask==2.1.0
- keras==2.11.0
- librosa==0.10.0
- matplotlib==3.7.1
- numpy==1.23.5
- pandas==1.5.3
- pygame==2.4.0
- scikit-learn==1.2.2
- tensorflow==2.11.0
- ...and more (refer to the complete list above for additional dependencies).

---

## Steps to Run the Project

### Step 1: Open the Project in VS Code

- Clone or download the project files.
- Open the project folder in Visual Studio Code.

### Step 2: Run the Main Script

- Locate the file named `bird.py` in the project directory.
- Run the script using the following command in the terminal:
    
    ```bash
    bash
    Copy code
    python bird.py
    
    ```
    

### Step 3: Use the GUI Interface

1. **Choose File**: Click the "Choose File" button in the GUI to select an audio file from the dataset.
2. **Convert to MFCC**: Click the "Convert to MFCC" button to process the audio file.
3. **Analyze MFCC**: Click the "Analyze MFCC" button to analyze the extracted features.
4. **Listen to Audio**: Click the "Mel-Spectrogram" image to listen to the processed audio.
5. **Identify Bird**: Click the bird image to view details about the identified bird species.

---

## Key Features

- **Audio Analysis**: Processes bird sound files into MFCC and mel-spectrogram formats for deep learning analysis.
- **Bird Identification**: Identifies the bird species based on the input audio file.
- **Interactive GUI**: Easy-to-use interface for audio processing and bird recognition.

---

## Notes

- Ensure all required libraries are installed before running the project.
- Use the provided dataset for optimal performance.
- For additional help, refer to the comments within the `bird.py` file.

---

Feel free to reach out if you encounter any issues while running the project. Enjoy exploring the fascinating world of bird sounds! 🎵🦜
