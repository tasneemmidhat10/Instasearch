import os
import requests
import json

# --- CONFIGURATION ---
# Load secrets from environment. Set NOTION_TOKEN and NOTION_DATABASE_ID
# in your shell or a .env file — never hard-code credentials.
NOTION_TOKEN = os.environ.get("NOTION_TOKEN")
DATABASE_ID = os.environ.get("NOTION_DATABASE_ID")

DEFAULTS = {
    'dataset':       'InstaDeepAI/ms_ninespecies_benchmark',
    'train_split':   'train',
    'val_split':     'validation',
    'test_split':    'test',
    'batch_size':    128,
    'd_model':       256,
    'n_heads':       4,
    'd_ff':          512,
    'n_layers':      2,
    'embed_dim':     64,
    'dropout':       0.2,
    'num_epochs':    25,
    'learning_rate': 1e-4,
    'weight_decay':  5e-3,
    'init_temp':     0.1,
    'warmup_epochs': 5,
    'output_dir':    './checkpoints',
    'save_every':    5,
    'seed':          42,
    'device':        None,
}

COLUMN_NAMES = {
    'dataset':       'Dataset',
    'train_split':   'Train Split',
    'val_split':     'Val Split',
    'test_split':    'Test Split',
    'batch_size':    'Batch Size',
    'd_model':       'Model Dim',
    'n_heads':       'Attention Heads',
    'd_ff':          'FF Dim',
    'n_layers':      'Layers',
    'embed_dim':     'Embed Dim',
    'dropout':       'Dropout',
    'num_epochs':    'Epochs',
    'learning_rate': 'Learning Rate',
    'weight_decay':  'Weight Decay',
    'init_temp':     'Init Temp',
    'warmup_epochs': 'Warmup Epochs',
    'output_dir':    'Output Dir',
    'save_every':    'Save Every',
    'seed':          'Seed',
    'device':        'Device',
}


def _make_property(value):
    if isinstance(value, bool):
        return {"rich_text": [{"text": {"content": str(value)}}]}
    if isinstance(value, (int, float)):
        return {"number": value}
    return {"rich_text": [{"text": {"content": str(value) if value is not None else "auto"}}]}


def log_experiment(
    run_name,
    accuracy,
    only_changed=False,
    # Data
    dataset=DEFAULTS["dataset"],
    train_split=DEFAULTS["train_split"],
    val_split=DEFAULTS["val_split"],
    test_split=DEFAULTS["test_split"],
    batch_size=DEFAULTS["batch_size"],
    # Model
    d_model=DEFAULTS["d_model"],
    n_heads=DEFAULTS["n_heads"],
    d_ff=DEFAULTS["d_ff"],
    n_layers=DEFAULTS["n_layers"],
    embed_dim=DEFAULTS["embed_dim"],
    dropout=DEFAULTS["dropout"],
    # Training
    num_epochs=DEFAULTS["num_epochs"],
    learning_rate=DEFAULTS["learning_rate"],
    weight_decay=DEFAULTS["weight_decay"],
    init_temp=DEFAULTS["init_temp"],
    warmup_epochs=DEFAULTS["warmup_epochs"],
    # Runtime
    output_dir=DEFAULTS["output_dir"],
    save_every=DEFAULTS["save_every"],
    seed=DEFAULTS["seed"],
    device=DEFAULTS["device"],
):
    """Log an experiment run to the Notion database.

    Pass only the hyperparameters you changed — set only_changed=True to
    automatically skip any param that still matches its default value.

    Example:
        log_experiment("run-01", accuracy=0.92, learning_rate=3e-4, only_changed=True)
    """
    hyperparams = {
        'dataset':       dataset,
        'train_split':   train_split,
        'val_split':     val_split,
        'test_split':    test_split,
        'batch_size':    batch_size,
        'd_model':       d_model,
        'n_heads':       n_heads,
        'd_ff':          d_ff,
        'n_layers':      n_layers,
        'embed_dim':     embed_dim,
        'dropout':       dropout,
        'num_epochs':    num_epochs,
        'learning_rate': learning_rate,
        'weight_decay':  weight_decay,
        'init_temp':     init_temp,
        'warmup_epochs': warmup_epochs,
        'output_dir':    output_dir,
        'save_every':    save_every,
        'seed':          seed,
        'device':        device,
    }

    if not NOTION_TOKEN or not DATABASE_ID:
        raise RuntimeError(
            "NOTION_TOKEN and NOTION_DATABASE_ID must be set in the environment."
        )

    headers = {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28",
    }

    properties = {
        "Run Name": {"title": [{"text": {"content": run_name}}]},
        "Accuracy": {"number": accuracy},
    }

    for key, value in hyperparams.items():
        if only_changed and DEFAULTS.get(key) == value:
            continue
        col = COLUMN_NAMES.get(key, key)
        properties[col] = _make_property(value)

    data = {"parent": {"database_id": DATABASE_ID}, "properties": properties}
    response = requests.post(
        "https://api.notion.com/v1/pages",
        headers=headers,
        data=json.dumps(data),
    )

    if response.status_code == 200:
        print("Successfully logged to Notion!")
    else:
        print(f"Error {response.status_code}: {response.text}")
