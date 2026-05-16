from huggingface_hub import snapshot_download

# Download Qwen — no approval needed
# print("Downloading Qwen2.5-7B-Instruct...")
# snapshot_download(repo_id="Qwen/Qwen2.5-7B-Instruct")
# print("Qwen download complete.")

# # Download Gemma — no approval needed
# print("Downloading Gemma-2-9B...")
# snapshot_download(repo_id="google/gemma-2-9b-it")
# print("Gemma download complete.")


print("Downloading Llama...")
snapshot_download(repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct")
print("Llama download complete.")