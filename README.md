
# Real-Time ASL Recognition Engine

An American Sign Laguange recongnition engine that works in real-time by using two models. One for recognizing hands and their shapes, and another custom convolutional neural network for recognizing which gesture corresponds to each letter.


## Authors

- [**@JummyJoeJackson**](https://github.com/JummyJoeJackson) (Handtracking)
- [**@danielkwan-dev**](https://github.com/danielkwan-dev) (ASL Recognition)
- [**@RatanotamKehal**](https://github.com/RatanotamKehal) (Game & Database)


## Features

- Real-time hand tracking with MediaPipe
- ASL alphabet recognition (A-Z, plus delete, space, nothing)
- Structured lessons with a learning path ranging from "The Basics" to "Advanced"
- User Profiles: Create accounts, save progress, track statistics via a local database using SQL
- The app automatically detects which letters you struggle with and focuses on them
- Earn XP for correct signs and maintain daily streaks to stay motivated
- Hold-to-confirm system for accurate letter input
- Build words and sentences by signing letters


## Project Structure

```
handtrack/
├── main.py                 # Main Game Application & Menu
├── user_manager.py         # User logic, stats, and database management
├── UserSetUp.py            # Database models (SQLAlchemy)
├── create_lessons.py       # Script to populate the curriculum
├── asl_database.db         # Local SQLite database (generated on run)
├── hand_landmarker.task    # MediaPipe hand model
├── requirements.txt
├── asl_model.pth           # Pre-trained model
└── src/
    ├── model.py            # Neural network architecture
    ├── train.py            # Model training
    ├── inference.py        # Standalone recognition app
    └── convert_images.py   # Dataset conversion utility
```
## Run Locally

#### Clone the project

```bash
  git clone https://github.com/JummyJoeJackson/handtrack.git
```

#### Go to the project directory

```bash
  cd handtrack
```

#### Install dependencies

```bash
  pip install -r requirements.txt
```

#### Run the Application or Run Simple Inference (No Game)

```bash
python main.py
```
or
```bash
python inference.py --model asl_model.pth
```

#### Options

```bash
python inference.py --model models/asl_model.pth --hold-time 1.5 --camera 0
```
- `--hold-time`: Seconds to hold pose before confirming (default: 2.0)
- `--camera`: Camera ID if you have multiple webcams (default: 0)





## Controls

- **Hold a sign** for 2 seconds to confirm the letter
- **Press `c`** (in inference mode) to clear text
- **Press `q`** to quit

## Collect Custom Data
In `python main.py` (regular mode):
Press letter keys (a-z) while showing hand signs to capture samples.
## Known Limitations

- **Letters J and Z**: These letters require hand movement to form (drawing a "J" or "Z" shape in the air). Since this system uses static hand poses, J and Z cannot be reliably recognized.

## Acknowledgements

 - [Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
 - [ASL Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet?resource=download)


## License

[MIT](https://choosealicense.com/licenses/mit/)

