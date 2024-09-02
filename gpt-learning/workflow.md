
**Andrej Karpathy** published a 4-hour-long video in July 2024, explaining how to reproduce a GPT model. It took me quite some time to finish the tutorial. The first reason is that I typed the code line by line, pausing the video every few minutes, which made the process slow. More importantly, there were many details to consider, and I had to read additional documents to understand them fully.

Finally, I had a trained 125M GPT model in my hands. I want to document my learning, coding, and training experience.

This article begins with my workflow for using a Lambda Labs GPU instance, and in future articles, I'll cover key insights I gained from the tutorial and the code.

## 1. Workflow for Using a Lambda Labs GPU Instance

Since I am mindful of costs, I aim to keep the GPU instance running for as little time as possible. The workflow achieves this by:
- Preprocessing training data on a Google Colab CPU instance.
- Using Hugging Face Hub to store training data and models, allowing fast data transfers to and from the GPU instance. Uploading or downloading data from my local PC would take hours of GPU instance time.
- Storing code on GitHub, enabling convenient code transfers to and from the GPU instance.

### Workflow
1. **Preprocess training data** on Google Colab CPU instance, then upload it to a Hugging Face Hub Dataset Repository.
2. **Develop training code** locally using Cursor IDE, and push it to GitHub.
3. **Train the model** on a Lambda Labs GPU instance:
   - Download the training code from GitHub to the Lambda Labs GPU instance.
   - Start training.
   - Upload the model (best checkpoint) to a Hugging Face Hub Model Repository.
   - Optionally, download other checkpoints to Google Drive from the Hugging Face Hub Model Repository.

### Prerequisites
- **Google Colab account**: For tokenizing the `edu_fineweb10B-sample`.
- **Google Drive account**: For saving the tokenized `edu_fineweb10B-sample`.
- **Hugging Face account**: For creating Hugging Face Hub dataset and model repositories.
- **Hugging Face Hub access token**: For uploading training data and models/checkpoints to the Hugging Face Hub repositories.
- **GitHub account**: For creating a code repository.
- **GitHub access token**: For pushing code to the GitHub repository.
- **Lambda Labs account**: For creating and using a Lambda Labs GPU instance.
- **Lambda Labs SSH key**: For accessing the Lambda Labs GPU instance from my local PC.

### Notes
- An SSH key is mandatory for using the Lambda Labs GPU instance. You must configure the SSH key on both your local PC and the GPU instance before use.
- I use access tokens for Hugging Face Hub and GitHub because it's easier to log in from the GPU instance with access tokens than configuring SSH.

## 1.1 Preprocess Training Data on Google Colab CPU Instance and Upload to Hugging Face Hub Dataset Repo

Andrej Karpathy downloaded and tokenized the `fineweb-edu-10B-sample` dataset on the fly during training on his 8x A100 GPU instance, which took about 30 minutes. To save money, I ran the tokenization on a Colab CPU instance. It took about 3 hours to tokenize and save the 100 `.npy` files to my Google Drive.

I experimented with different ways to make the files available to the Lambda Labs GPU instance, such as Google Drive and Hugging Face Hub. I found that Hugging Face Hub is the best option because it's easy to download files to the Lambda Labs instance.

For details on the approaches I tried, see Notebook xxxx.

### Current Approach

#### Step 1: Tokenize and Save

- **Tokenize the `fineweb-edu-10B-sample`** on a Google Colab CPU instance using Andrej Karpathy's `fineweb.py` script with minor modifications.
- `fineweb-edu/sample/10B` is a subset of `fineweb`, containing 10B tokens.
- The tokenization process took about 3 hours on the Colab CPU instance. A CPU instance is sufficient since the process involves iterating over all sentences, which doesn't require a GPU.
- The generated `.npy` files are saved to a Google Drive folder. This is because it's faster than saving to a local PC hard drive, and the 107GB Colab hard drive is just enough to load and process the dataset but not to save the tokenized files.
- 100 `.npy` files were generated, each containing 100M tokens (using `uint16`), and they are about 170MB each.

#### Step 2: Upload to Hugging Face Hub

The easiest way to upload `.npy` files to Hugging Face Hub is to use `huggingface_hub.HfApi`.

- **Create a Dataset Repository** using `HfApi`. Ensure you specify `repo_type="dataset"` to avoid errors.
- Use `HfApi.upload_file()` to upload the files.

**Example Code:**
```python
import os
from huggingface_hub import HfApi, login

api = HfApi()
login(token="")  # Fill in your Hugging Face token

local_dir = "/content/drive/MyDrive/Colab Notebooks/nanogpt/edu_fineweb10B/"
repo_id = "jfzhang/edu_fineweb10B_tokens_npy_files"

api.create_repo(repo_id=repo_id, repo_type="dataset")

fn_list = os.listdir(local_dir)
for filename in fn_list:
    if filename.endswith(".npy"):
        local_path = os.path.join(local_dir, filename)
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="dataset"
        )
```

#### Step 3: Download the Dataset in the Training Script

During training, download the tokenized files from the Hugging Face Hub Dataset repo to the Lambda Labs instance using `snapshot_download()`.

**Example Code:**
```python
from huggingface_hub import snapshot_download

repo_id = "jfzhang/edu_fineweb10B_tokens_npy_files"
local_dir = "./edu_fineweb10B/"
snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=local_dir)
```

### Advantages
- You can use `.npy` files directly, avoiding the need to convert to other formats.
- File upload/download is very fast.
- Authentication is straightforward and reliable.

## 1.2 Develop Training Code on Cursor IDE and Push to GitHub

Cursor IDE is an excellent AI assistant for writing, explaining, and debugging code. Configuring GitHub on Cursor IDE is the same as in VSCode.

## 1.3 Train the Model on Lambda Labs GPU Instance and Upload to Hugging Face Hub Model Repo

#### Step 1: Create a Hugging Face Hub Model Repository

Treat the Hugging Face Hub model repository like a Git repository. Create the repository using the Hugging Face Hub Web UI, then use Git commands in the terminal (SSH into the Lambda Labs instance) to push/pull/clone the repository as needed.

#### Step 2: Start a Lambda Labs GPU Instance

Start the instance from the Lambda Labs Web UI. It takes a few minutes for the instance to be ready for SSH access.

#### Step 3: SSH into the GPU Instance

Use a terminal on your local PC (e.g., Windows PowerShell).

```bash
ssh -i <path-to-private-key> ubuntu@<public-ip-address>
```
Replace `<path-to-private-key>` with the path to your private key file and `<public-ip-address>` with the instance's public IP address.

#### Step 4: Clone the GitHub Repo

If the repository is public and no modifications need to be saved, use:
```bash
git clone https://github.com/<username>/<repo-name>.git
```
For private repositories or when saving changes:
```bash
git clone https://<username>:<access-token>@github.com/<username>/<repo-name>.git
```

#### Step 5: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 6: Start Training

Train on one node with 8 GPUs:
```bash
torchrun --standalone --nproc_per_node=8 <training-script>
```
This training took less than 3 hours. It was exciting to watch the training log in real time, especially since it was my first time using DDP and training a model with 8 GPUs.

#### Step 7: Clone the Hugging Face Hub Model Repository

While training, clone the Hugging Face Hub model repository in another terminal session. This repository will be used to upload the best or final checkpoints.

```bash
git clone https://<huggingface_username>:<huggingface_access_token>@huggingface.co/<huggingface_username>/<model-repo-name>
sudo apt-get install git-lfs
git lfs install
git config --global user.email "<your-email>"
git config --global user.name "<your-username>"
```

#### Step 8: Upload Checkpoints to Hugging Face Hub Model Repo

Copy the selected checkpoints to the model repository's local folder.
```bash
cp <checkpoint-path> <model-repo-path>/<checkpoint-name>
```
Then push the checkpoints:
```bash
cd <model-repo-path>
git add .
git commit -m "add checkpoint"
git push
```

#### Step 9 (Optional): Download Checkpoints to Google Drive

All the data on Lambdalabs instance will be deleted after the instance is terminated. If you want to keep the checkpoints other than the ones uploaded to Huggingface-Hub, you can download them to Google Drive like me.
1. Start a Google Colab CPU notebook. 
2. mount Google Drive
3. Copy the ssh private key file used by Lambdalabs instance to the Colab instance at a path like ~/.ssh/<private-key-filename> 
4. scp to download the checkpoints from GPU instance to Google Drive
```bash
!chmod 400 ~/.ssh/<private-key-filename>
!scp -o StrictHostKeyChecking=no -i ~/.ssh/<private-key-filename> ubuntu@<gpu-instance-public-ip-address>:~/<checkpoint-path-on-gpu-instance> /content/drive/MyDrive/<path-on-google-drive>
```
