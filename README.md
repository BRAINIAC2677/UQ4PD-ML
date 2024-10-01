# UQParkNet | `U`ncertainty `Q`uantified `Park`inson `Net`work

## Insights
- In all three unimodal models, the BNN variants have more or less same accuracy as their inference time mc-dropout counterpart.
- However, the BNN variant of fusion network drops accuracy by almost 0.08 both on BNN, inference time mc-dropout unimodal models.
- My hypothesis is that it will catch up after hyper-parameter tuning of the BNN variant of the fusion network because the dev accuracy also dropped drastically in BNN variant of fusion network.

## File Descriptions

### Data
- `all_file_user_metadata.csv` -- contains demographic information for all participants who recorded at least one task.
- `all_task_ids.txt` -- contains the unique ids of 845 participants completing all three tasks
- `demographic_details.csv` -- demographic information for 845 participants completing all three tasks
- `dev_set_participants.txt` -- unique ids of participants randomly selected for the validation set
- `test_set_participants.txt` -- unique ids of participants randomly selected for the test set
- `facial_expression_smile/facial_dataset.csv` -- extracted features for the facial expression (smile) task
- `finger_tapping/features_demography_diagnosis_Nov22_2023.csv` -- extracted features for the finger-tapping task
- `quick_brown_fox/wavlam_fox_features.csv` -- extracted features for the speech task

### Code
- `unimodal_scripts/{task_name}/constants.py` -- paths set up to run task-specific model.
- `unimodal_scripts/{task_name}/models.py` -- model architecture code for the specific task.
- `unimodal_scripts/{task_name}/unimodal_{task_tag}.py` -- train, dev and test code for the tasks-specific model.
- `fusion_scripts/constants.py` -- paths set up to run the experiment
- `fusion_scripts/uncertainty_aware_fusion.py` -- train, dev and test code for fusion network.

## Setup

### Setup Instructions for Linux/macOS

- Clone this repository.
    ```
    git clone https://github.com/BRAINIAC2677/uqparknet.git
    ```
- Move to root of the project.
    ```
    cd uqparknet
    ```

- Create `saved_models` directory for saving the trained models.
    ```
    mkdir saved_models
    ```

- Set the Base Directory `base=$(pwd)`

- #### Using Python's venv

    - Create a New Virtual Environment
        ```
        python -m venv $base/uqparknet
        ```

    - Activate the Virtual Environment
        ```
        source $base/uqparknet/bin/activate
        ```

    - Install Required Packages
        ```
        pip install -r $base/requirements.txt
        ```

- #### Using Conda

    - Create a New Conda Environment
        ```
        conda create --prefix $base/uqparknet python=3.10.14
        ```

    - Activate the Conda Environment
        ```
        conda activate $base/uqparknet
        ```

    - Install Required Packages
        ```
        pip install -r $base/requirements.txt
        ```

## Running the Models 

### Unimodal Models

- #### Smile Model
    ```
    cd $base/code/unimodal_scripts/facial_expression_smile
    python unimodal_smile.py --model ShallowBNN
    ```

- #### Finger Tapping Model
    ```
    cd $base/code/unimodal_scripts/finger_tapping
    python unimodal_finger.py --model ShallowANN
    ```

- #### Speech Model
    ```
    cd $base/code/unimodal_scripts/quick_brown_fox
    python unimodal_fox.py --model BNN
    ```

### Fusion Model

```
cd $base/code/fusion_scripts
python uncertainty_aware_fusion.py --fusion_model bayesian
```

## Contact
For any queries contact,
- Asif Azad - asifazad0178@gmail.com

