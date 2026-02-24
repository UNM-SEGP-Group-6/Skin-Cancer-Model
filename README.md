# Skin Cancer Classification Model
Multimodal approach towards classifying skin cancer using transfer learning with clinical metadata.
[WIP]

## Prerequisites:
- **Python** 3.12
-  **pip** package manager

    ```
    pip install -r requirements.txt
    ```
    ### Troubleshooting:
    - If you receive a Protobuf gencode version error try running the following command or modify it to install the desired version.
    
        ```
        pip install --upgrade protobuf
        OR
        pip install protobuf==[desired_version_number]
        ```
    - Various errors can take place due to the images becoming corrupted at any point (e.g. Preprocessing, Extracting, Training, etc), in which case, it is advised to store the .zip file of the dataset in a separate location and extract it again.