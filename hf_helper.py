import sys
import argparse
import json
from huggingface_hub import HfApi, login

def handle_whoami(args):
    token = args.token
    if not token:
        print(json.dumps({"success": False, "error": "Token is required"}))
        return
    
    try:
        api = HfApi(token=token)
        user_info = api.whoami()
        # Extract clean profile info
        profile = {
            "success": True,
            "username": user_info.get("name"),
            "fullname": user_info.get("fullname"),
            "email": user_info.get("email"),
            "avatarUrl": user_info.get("avatarUrl"),
            "orgs": [org.get("name") for org in user_info.get("orgs", [])],
            "authType": user_info.get("auth", {}).get("type", "unknown")
        }
        print(json.dumps(profile))
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))

def handle_search_models(args):
    query = args.query
    limit = args.limit or 10
    token = args.token
    
    try:
        api = HfApi(token=token) if token else HfApi()
        models = api.list_models(search=query, limit=limit, sort="downloads", direction=-1)
        results = []
        for m in models:
            results.append({
                "id": m.id,
                "author": m.author,
                "downloads": getattr(m, "downloads", 0),
                "likes": getattr(m, "likes", 0),
                "lastModified": str(m.lastModified) if getattr(m, "lastModified", None) else None,
                "pipeline_tag": getattr(m, "pipeline_tag", None)
            })
        print(json.dumps({"success": True, "models": results}))
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))

def handle_search_datasets(args):
    query = args.query
    limit = args.limit or 10
    token = args.token
    
    try:
        api = HfApi(token=token) if token else HfApi()
        datasets = api.list_datasets(search=query, limit=limit, sort="downloads", direction=-1)
        results = []
        for d in datasets:
            results.append({
                "id": d.id,
                "author": d.author,
                "downloads": getattr(d, "downloads", 0),
                "likes": getattr(d, "likes", 0),
                "lastModified": str(d.lastModified) if getattr(d, "lastModified", None) else None
            })
        print(json.dumps({"success": True, "datasets": results}))
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))

def handle_upload_model(args):
    token = args.token
    repo_id = args.repo_id
    folder = args.folder
    
    if not token or not repo_id or not folder:
        print(json.dumps({"success": False, "error": "Token, repo-id, and folder are required"}))
        return
    
    try:
        api = HfApi(token=token)
        # Create repository if it doesn't exist
        print(f"Creating repository if it doesn't exist: {repo_id}", file=sys.stderr)
        api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")
        
        # Upload folder
        print(f"Uploading folder {folder} to {repo_id}", file=sys.stderr)
        api.upload_folder(
            folder_path=folder,
            repo_id=repo_id,
            repo_type="model"
        )
        print(json.dumps({"success": True, "repoUrl": f"https://huggingface.co/{repo_id}"}))
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))

def main():
    parser = argparse.ArgumentParser(description="Hugging Face Helper CLI")
    subparsers = parser.add_subparsers(dest="command", help="Subcommand to run")
    
    # whoami subcommand
    parser_whoami = subparsers.add_parser("whoami")
    parser_whoami.add_argument("--token", required=True, help="HF Access Token")
    
    # search-models subcommand
    parser_sm = subparsers.add_parser("search-models")
    parser_sm.add_argument("--query", required=True, help="Search query")
    parser_sm.add_argument("--limit", type=int, default=10, help="Max results")
    parser_sm.add_argument("--token", help="Optional HF Token")
    
    # search-datasets subcommand
    parser_sd = subparsers.add_parser("search-datasets")
    parser_sd.add_argument("--query", required=True, help="Search query")
    parser_sd.add_argument("--limit", type=int, default=10, help="Max results")
    parser_sd.add_argument("--token", help="Optional HF Token")
    
    # upload-model subcommand
    parser_up = subparsers.add_parser("upload-model")
    parser_up.add_argument("--repo-id", required=True, help="Target Hugging Face Repo ID (e.g. username/model-name)")
    parser_up.add_argument("--folder", required=True, help="Local path to model folder")
    parser_up.add_argument("--token", required=True, help="HF Token with WRITE access")
    
    args = parser.parse_args()
    
    if args.command == "whoami":
        handle_whoami(args)
    elif args.command == "search-models":
        handle_search_models(args)
    elif args.command == "search-datasets":
        handle_search_datasets(args)
    elif args.command == "upload-model":
        handle_upload_model(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
