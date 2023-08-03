# AI-DINO-GAME

AI-DINO-GAME is a Python project that utilizes a neural network to play the Google Chrome Dino Game. The neural network is trained to scan the screen and make decisions to control the dino's actions, allowing it to learn and improve its performance over time.

<!-- ![Dino Game Screenshot](screenshot.png) -->

## Table of Contents

- [AI-DINO-GAME](#ai-dino-game)
  - [Table of Contents](#table-of-contents)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
  - [Usage](#usage)
  - [Training](#training)
  - [License](#license)
  - [Acknowledgement](#acknowledgement)

## Getting Started

### Prerequisites

Before running the AI-DINO-GAME project, make sure you have the following prerequisites:

- Python (>=3.11)
- Libraries listed in `requirements.txt`

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/jsmatias/AI-DINO-GAME.git
   ```

2. Navigate to the project directory:

   ```bash
   cd AI-DINO-GAME
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Use the set-environment.ipynb to dynamically set the region of the screen to be scanned.

2. Use the main.ipynb to run the main loop with the settings predefined on the step above.

3. Watch as the neural network learns to play the Dino Game in your Google Chrome browser.

## Training

The neural network is trained using a combination of reinforcement learning techniques. The module `environment.py` takes screen shots, prepare the data, and control the keyboard to interaction with the dino game. The module `agent.py` contains the neural network and is responsible for the learning process as well as predicting the best actions based on image analysis. 

<!-- For more details on the training process, algorithm, and techniques used, refer to the [TRAINING.md](TRAINING.md) file. -->
<!-- 
## Contributing

Contributions to AI-DINO-GAME are welcome! If you find any issues or have improvements to suggest, feel free to open an issue or submit a pull request. -->

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgement
This code was based and adapted from [SaralTayal123](https://github.com/SaralTayal123/ChromeDinoAI)