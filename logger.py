import argparse
import requests
import json

# --- CONFIGURATION ---
NOTION_TOKEN = "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"
DATABASE_ID = "3438c6deff278098aeec000ca501aed1"

def log_to_notion(run_name, lr, batch_size, epochs, accuracy):
    url = "https://api.notion.com/v1/pages"
    
    headers = {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28" # Or current 2026 version
    }

    # This dictionary maps your script arguments to Notion columns
    data = {
        "parent": {"database_id": DATABASE_ID},
        "properties": {
            "Run Name": {
                "title": [{"text": {"content": run_name}}]
            },
            "Learning Rate": {
                "number": lr
            },
            "Batch Size": {
                "number": batch_size
            },
            "Epochs": {
                "number": epochs
            },
            "Accuracy": {
                "number": accuracy
            }
        }
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    
    if response.status_code == 200:
        print("Successfully logged to Notion! 🚀")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Log ML Experiments to Notion")
    
    # Add your hyperparameters here
    parser.add_argument("--name", type=str, required=True, help="Name of the run")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--acc", type=float, required=True, help="Final accuracy")

    args = parser.parse_args()

    log_to_notion(args.name, args.lr, args.batch, args.epochs, args.acc)