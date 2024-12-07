# AI-based modeling of breathing trajectories in particle therapy

## Abstract
This project develops a Long-Short Term Memory (LSTM) model to predict tumor motion in real-time, enabling dynamic adaptation of particle therapy to respiratory motion. The model ensures precise radiation dose targeting despite variations caused by patient breathing, particularly for lung tumors. Trained on data from experimental carbon ion therapy at Centro Nazionale di Adrotherapia Oncologia, the model achieves high prediction accuracy even with irregular motion.

## Installation
1. Download the repository to a local folder of your preference or clone the repository.
2. Install the required Python libraries:
```bash
pip install -r requirements.txt
```
## Usage

### Data Generation
The sine( ) and sine2( ) functions generate synthetic breathing motion data, simulating patient respiratory patterns. You can modify the parameters to match actual breathing curves.

### Model Training 
The model is based on an LSTM architecture to predict tumor motion. You can train the model with your own data using the run_pred( ) function.
