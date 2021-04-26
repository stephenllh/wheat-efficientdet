# <span>Bengali.AI</span> Handwritten Grapheme Classification Bronze Medal Solution

<!-- ABOUT THE PROJECT -->
## About The Project

<br/>
<p align="center">
  <img src="/image/image.png" alt="Competition image"/>
</p>


<!-- ![Product Name Screen Shot](/image/image.png) -->

This is my solution to the [Global Wheat Detection](https://www.kaggle.com/c/global-wheat-detection) that uses EfficientDet.

<br/>

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running, follow these simple example steps.
<br/><br/>

### Prerequisites

* PyTorch (version 1.6.0)

  Install using Anaconda:
  ```sh
  conda install pytorch=1.6.0 -c pytorch
  ```

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/stephenllh/wheat_efficientdet.git
   ```

1. Change directory
   ```sh
   cd wheat_efficientdet
   ```

2. Install packages
   ```sh
   pip install requirements.txt
   ```
<br/>

<!-- USAGE EXAMPLES -->
## Usage

1. Create a directory called `input`
   ```sh
   mkdir input
   cd input
   ```

2. Download the dataset into the folder
    - Option 1: Use Kaggle API
      - `pip install kaggle`
      - `kaggle competitions download -c global-wheat-detection`
    - Option 2: Download the dataset from the [competition website](https://www.kaggle.com/c/global-wheat-detection/data).

3. Run the training script
   ```sh
   cd ..
   python train.py
   ```

4. (Optional) Run the inference script
   ```sh
   python inference.py
   ```

<br/>


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.
<br></br>


<!-- CONTACT -->
## Contact

Stephen Lau - [Email](stephenlaulh@gmail.com) - [Twitter](https://twitter.com/StephenLLH) - [Kaggle](https://www.kaggle.com/faraksuli)
