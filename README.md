# Motion Subtitle Generator

## Overview
The Motion Subtitle Generator is an AI-based system that analyzes visual motion in video content to automatically generate **time-synchronized, action-based subtitles**. Unlike traditional subtitles that focus only on spoken dialogue, this project aims to improve accessibility and contextual understanding by describing visible actions and movements.

This project is developed as an **academic and research-oriented system** and is under continuous improvement.

---

## Motivation
Many videos rely heavily on visual actions to convey meaning, especially in silent, action-driven, or instructional content. Manually annotating such videos is time-consuming and inefficient. This project explores how Artificial Intelligence and motion analysis can be applied to automatically generate meaningful action-based subtitles.

---

## Key Features
- Automatic generation of action-based subtitles  
- Motion analysis from video frames  
- Time-synchronized subtitle output  
- Supports standard subtitle formats (SRT, VTT)  
- Modular and extensible design  

---

## System Workflow
1. Input video is processed frame by frame  
2. Motion patterns are analyzed to identify significant actions  
3. Detected actions are mapped to descriptive text  
4. Subtitles are generated with accurate timestamps  
5. Output is saved in standard subtitle formats  

---

## Technologies Used
- **Programming Language:** Python  
- **AI / ML Concepts:** Motion analysis, pattern recognition, model integration  
- **Video Processing:** Frame extraction and motion comparison  
- **Output Formats:** `.srt`, `.vtt`  

---

## Repository Structure
motion-subtitle-generator/
│
├── quick_motion_subtitles.py # Main processing script
├── srt_to_vtt.py # Subtitle format conversion
├── player_overlay.html # Frontend subtitle overlay
│
├── samples/
│ ├── sample_input.mp4 # Sample input video
│ ├── sample.srt # Generated subtitle output
│ └── sample.vtt
│
├── README.md
├── requirements.txt
└── LICENSE


---

## How to Run the Project

### Prerequisites
- Python 3.x installed  
- Basic knowledge of running Python scripts  
- Required Python libraries installed  

---

### Step 1: Clone the Repository
git clone https://github.com/<your-username>/motion-subtitle-generator.git
cd motion-subtitle-generator

**Step 2: Install Dependencies**
pip install -r requirements.txt

**Step 3: Prepare Sample Video**
Create a folder named samples/ (if not already present).
Add a sample video file to the folder.
Rename the video file to:
'sample_input.mp4'
Example:
samples/
└── sample_input.mp4

**Step 4: Run the Motion Subtitle Generator**
Execute the main script:
python quick_motion_subtitles.py
The script will:
Process the input video
Analyze motion and actions
Generate subtitle files

**Step 5: Check Generated Subtitle Files**
After execution, the following files will be generated:
sample.srt – Action-based subtitle file
sample.vtt – Web-compatible subtitle file
These files demonstrate the output of the system.

**Step 6: Convert Subtitle Format (Optional)**
If required, convert subtitles manually using:
python srt_to_vtt.py

**Step 7: View Subtitles Using Frontend Overlay**
Open player_overlay.html in a web browser.
Load the sample video and the corresponding subtitle file.
Subtitles will appear synchronized with the video playback.
This step helps visualize the generated subtitles in a real-world player environment.

---

### Project Status
This project is under active development. Current efforts focus on:
Improving action detection accuracy
Reducing false motion detections
Enhancing subtitle alignment and clarity
Improving robustness across different video types
