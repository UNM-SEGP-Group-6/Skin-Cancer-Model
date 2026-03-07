# Skin Cancer Classification Model
Multimodal approach towards classifying skin cancer using transfer learning with clinical metadata. [WIP]

## Prerequisites:
- **Python** 3.12
-  **PIP** Package Manager
- All the required packages are listed in the requirements.txt file.

## Installation Steps:
- Clone the repository:
    ```
    git clone [INSERT GITHUB REPO LINK]
    cd skin-cancer-classification
    ```
    ---
    ### Windows/CMD Prompt
    - Create a virtual environment to avoid dependency clashes:
        ```
        python -m venv venv
        ```
    - Activate the virtual environment:
        ```
        venv\Scripts\activate
        ```
    - Install the required packages:
        ```
        pip install -r requirements.txt
        ```
    ---
    ### IDE
    - Open the project in your preferred IDE
    - Select the kernel/Environments option in the top right corner of the IDE.
    [![image-1.png](https://i.postimg.cc/J7knpKrn/image-1.png)](https://postimg.cc/Dm7nZrbk)
    - Select Another Kernel OR "Python Environments"
    - Create Python Environment
    - Venv (And proceed with the rest of the installation steps)
        - Note:
            This step is recommended if you are using the Notebook during the prototyping steps.

## Tensorboard & WANDB
- To test Tensorboard:
    ```
    tensorboard --logdir "c:\Insert\Runs\Path\Here" --port 6070
    ```
- To test WANDB:
    ```
    wandb login
    ```

## Troubleshooting:
1. If you receive a Protobuf gencode version error try running the following command or modify it to install the desired version.
    
    ```
    pip install --upgrade protobuf
    OR
    pip install protobuf==[desired_version_number]
    ```
2. Various errors can take place due to the images becoming corrupted at any point (e.g. Preprocessing, Extracting, Training, etc), in which case, it is advised to store the .zip file of the dataset in a separate location and extract it again.
