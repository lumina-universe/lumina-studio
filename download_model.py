import sys
import os
import argparse
import requests
import time
from urllib.parse import urlparse

def format_size(bytes_size):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"

def download_from_url(url, output_dir, filename=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    parsed_url = urlparse(url)
    if not filename:
        filename = os.path.basename(parsed_url.path)
        if not filename:
            filename = "downloaded_model.bin"
            
    target_path = os.path.join(output_dir, filename)
    print(f"INFO: Starting download from URL: {url}", flush=True)
    print(f"INFO: Saving to: {target_path}", flush=True)
    
    start_time = time.time()
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    
    last_report_time = time.time()
    last_downloaded = 0
    
    with open(target_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024): # 1MB chunks
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                
                current_time = time.time()
                elapsed = current_time - last_report_time
                if elapsed >= 1.0 or downloaded == total_size:
                    # Calculate speed
                    speed = (downloaded - last_downloaded) / elapsed if elapsed > 0 else 0
                    speed_str = f"{format_size(speed)}/s"
                    
                    # Calculate percentage
                    percent = (downloaded / total_size) * 100 if total_size > 0 else 0
                    percent_str = f"{percent:.1f}%" if total_size > 0 else "N/A"
                    
                    # Print progress in parsed format
                    print(f"PROGRESS:{percent_str} | SPEED:{speed_str} | DOWNLOADED:{format_size(downloaded)} / {format_size(total_size)}", flush=True)
                    
                    last_report_time = current_time
                    last_downloaded = downloaded
                    
    total_elapsed = time.time() - start_time
    average_speed = downloaded / total_elapsed if total_elapsed > 0 else 0
    print(f"SUCCESS: Download finished. Total size: {format_size(downloaded)}. Average speed: {format_size(average_speed)}/s. Saved to {target_path}", flush=True)
    return target_path

def download_from_hf(repo_id, output_dir, token=None, filename=None):
    from huggingface_hub import snapshot_download, hf_hub_download
    
    print(f"INFO: Starting download from Hugging Face: {repo_id}", flush=True)
    
    if filename:
        print(f"INFO: Downloading single file: {filename}", flush=True)
        # Download a single file
        target_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=output_dir,
            token=token,
            local_dir_use_symlinks=False
        )
        print(f"SUCCESS: Download finished. Saved to {target_path}", flush=True)
        return target_path
    else:
        print(f"INFO: Downloading entire repository snapshot", flush=True)
        # Download snapshot
        target_path = snapshot_download(
            repo_id=repo_id,
            local_dir=output_dir,
            token=token,
            local_dir_use_symlinks=False,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"] # Skip weights format we don't need
        )
        print(f"SUCCESS: Snapshot download finished. Directory: {target_path}", flush=True)
        return target_path

def download_from_modelscope(repo_id, output_dir, filename=None):
    from modelscope.hub.snapshot_download import snapshot_download
    print(f"INFO: Starting download from ModelScope: {repo_id}", flush=True)
    if filename:
        print(f"WARNING: ModelScope single-file downloads are managed through SDK cache. Downloading snapshot instead.", flush=True)
    target_path = snapshot_download(
        model_id=repo_id,
        cache_dir=output_dir,
        is_strictly_trusted=False
    )
    print(f"SUCCESS: Snapshot download finished. Saved to {target_path}", flush=True)
    return target_path

def main():
    parser = argparse.ArgumentParser(description="Model Downloader utility")
    parser.add_argument("--source", choices=["url", "huggingface", "modelscope"], required=True, help="Download source")
    parser.add_argument("--repo-id", help="Repository ID (e.g. facebook/opt-125m or llm-research/Meta-Llama-3-8B-Instruct)")
    parser.add_argument("--url", help="HTTP/HTTPS URL of the model file")
    parser.add_argument("--output-dir", required=True, help="Output directory path")
    parser.add_argument("--filename", help="Optional specific filename to download (HF single file or HTTP custom name)")
    parser.add_argument("--token", help="Optional Hugging Face access token")
    
    args = parser.parse_args()
    
    try:
        if args.source == "url":
            if not args.url:
                raise ValueError("--url is required when --source is 'url'")
            download_from_url(args.url, args.output_dir, args.filename)
        elif args.source == "huggingface":
            if not args.repo_id:
                raise ValueError("--repo-id is required when --source is 'huggingface'")
            download_from_hf(args.repo_id, args.output_dir, args.token, args.filename)
        elif args.source == "modelscope":
            if not args.repo_id:
                raise ValueError("--repo-id is required when --source is 'modelscope'")
            download_from_modelscope(args.repo_id, args.output_dir, args.filename)
    except Exception as e:
        print(f"FAILURE: Download failed. Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

