# CNN
CNN Model to be trained on the MRA-MIDAS dataset.

MRA-MIDAS Dataset Source: https://doi.org/10.71718/15nz-jv40

# Prerequisites:
- **Python** 3.12
-  **pip** package manager
    ```
    pip (Check if pip is installed/cd'd to the correct directory)
    pip install --upgrade pip
    python.exe -m pip install --upgrade pip
    pip install tensorflow
    ```
    ### Troubleshooting:
    - If you receive a Protobuf gencode version error try running the following command or modify it to install the desired version.
    ```
    pip install --upgrade protobuf
    OR
    pip install protobuf==[desired_version_number]
    ```

# Dataset Instructions:
1. Acquire the Dataset link from [here](https://aimi.stanford.edu/datasets/mra-midas-Multimodal-Image-Dataset-for-AI-based-Skin-Cancer).
2. Download AZCopy from [here](https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10?tabs=windows).
3. Open Command Prompt as Administrator.
4. Navigate to the directory where AZCopy is installed.
5. Run the following command: 
    ```
    azcopy login
    ```
6. You'll be prompted to visit [https://microsoft.com/devicelogin](https://microsoft.com/devicelogin) and enter a unique code to authenticate.
7. Run the following command: 
    ```
    azcopy copy "MRA-MIDAS Link" "C:Destination/Local/Path" --recursive=true.
    ```
8. It should output the following: 
    ```
    INFO: Scanning...
    INFO: Any empty folders will not be processed, because source and/or destination doesn't have full folder support

    Job [UniqueID] has started
    Log file is located at: C:\Destination/Local/Path/UniqueID.log
    ```
9. Once all the files are downloaded you should receive the following output:
    ```
    100.0 %, 3418 Done, 0 Failed, 0 Pending, 0 Skipped, 3418 Total,                                   


    Job [UniqueID] summary
    Elapsed Time (Minutes): [Time]
    Number of File Transfers: 3418
    Number of Folder Property Transfers: 0
    Number of Symlink Transfers: 0
    Total Number of Transfers: 3418
    Number of File Transfers Completed: 3418
    Number of Folder Transfers Completed: 0
    Number of File Transfers Failed: 0
    Number of Folder Transfers Failed: 0
    Number of File Transfers Skipped: 0
    Number of Folder Transfers Skipped: 0
    Number of Symbolic Links Skipped: 0
    Number of Hardlinks Converted: 0
    Number of Special Files Skipped: 0
    Total Number of Bytes Transferred: 3807825722
    Final Job Status: Completed
    ```

    ### References & Guides:
    - https://aimi.stanford.edu/datasets/mra-midas-Multimodal-Image-Dataset-for-AI-based-Skin-Cancer
    - https://learn.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10

    ### Troubleshooting:
    - If the AIMI link doesn't load or proceed to the next step try using a different browser due to encountered issues with Microsoft Edge.