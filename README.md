# Hip and Knee Joints 2D Angles Estimation and Video Processing

This project calculates 2D knee and hip joint angles from a video file using the Mediapipe library and other Python tools. The results are displayed on the processed video and saved as .CSV, images, and a video file. The tool is flexible for either the left or right side of the body and offers options to visualize the processing in real-time.

## Table of Contents
1. [Installation](#installation)
2. [Project Structure](#project-structure)
3. [Usage](#usage)
    - [Basic Command](#basic-command)
    - [Script Arguments](#script-arguments)
4. [Example](#example)
5. [License](#license)

## Installation

### 1. Clone the Repository
Start by cloning this repository to your local machine:

```bash
git clone https://github.com/your-username/mediapipe2dangle.git
cd mediapipe2dangle
```

### 2. Create a Virtual Environment
It’s recommended to create a virtual environment to manage the dependencies:

```bash
python -m venv env
```
Or
```bash
conda create -n mediapipe2dangle python=3.9
```

Activate the virtual environment:

- On **Windows**:
  ```bash
  .\env\Scripts\activate
  ```
- On **macOS/Linux**:
  ```bash
  source env/bin/activate
  ```
- Usind conda:
  ```bash
  conda activate mediapipe2dangle
  ```

### 3. Install Dependencies
Install the required libraries using `pip`:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file contains the following packages:
- `opencv-python`
- `mediapipe`
- `matplotlib`
- `numpy`
- `pandas`
- `Pillow`

Make sure your environment is set up with the correct versions of these dependencies.

## Project Structure

```bash
mediapipe2dangle/
│
├── videos/               # Directory where video files are stored
├── main_run.py           # Main script for running the joint angle estimation
├── maintools.py          # Helper functions used by the main script
├── example.py            # Example script demonstrating usage (.py file)
├── example.ipynb         # Jupyter notebook example (.ipynb file)
├── README.md             # Documentation
├── requirements.txt      # Python dependencies
└── videos/results/       # Output folder (created automatically)
```

## Usage

To run the joint angle estimation tool, use the `main_run.py` script. The script processes a video and outputs a video with overlaid angles, a CSV file with the angles over time, and a plot image.

### Basic Command

```bash
python main_run.py --videopath path_to_your_video.mp4 --side r
```

This will process the right side (`r`) of the body by default. Change `r` to `l` to process the left side.

### Script Arguments

- `--videopath` (required): Path to the video file.
- `--side` (optional): Body side to analyze, either 'r' (right) or 'l' (left). Default is 'r'.
- `--save` (optional): Whether to save results. Use `True` or `False`. Default is `True`.
- `--show` (optional): Whether to display the video processing in real-time. Use `True` or `False`. Default is `False`.
- `--min_confidence` (optional): Confidence threshold for Mediapipe tracking. Default is `0.9`.
- `--scale_factor` (optional): Scale factor for resizing the graph. Default is `0.5`.

### Full Example

```bash
python main_run.py --videopath ./videos/example.mp4 --side r --save True --show True --min_confidence 0.8 --scale_factor 0.7
```

This command processes the video `example.mp4` located in the `videos` directory, calculating angles on the right side, saving the output, showing the video while processing, with a minimum confidence of `0.8` and graph scaling at `70%`.

## Example Output

When running the script, the following outputs will be generated in the `results/` directory:
1. **Video with overlaid angles** (`.mp4`)
2. **CSV file** containing frame numbers and corresponding knee and hip angles.
3. **Image plot** showing the knee and hip angles over time.

Example of CSV output:

| Frame | Knee_Angle | Hip_Angle |
|-------|------------|-----------|
| 0     | 145.3      | 150.2     |
| 1     | 146.1      | 149.9     |
| ...   | ...        | ...       |

Example plot of angles:

![Knee and Hip Angles Plot](results/example_markerless_r.jpg)

## License
This project is primarily licensed under the GNU Lesser General Public License v3.0. Note that the software is provided "as is", without warranty of any kind, express or implied. 

## Funding
This project was partially financed by the Dean’s Office for Research and Innovation of the University of São Paulo - Support to New Professors.
