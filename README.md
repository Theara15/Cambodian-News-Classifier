# Cambodian News Classifier

This repository contains a Cambodia news classification Streamlit app built with PyTorch and HuggingFace models.

## Run locally

```powershell
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

## Deploy to Streamlit Cloud

1. Push this repository to GitHub.
2. In Streamlit Cloud, create a new app and connect the repo.
3. Set the app entrypoint to `streamlit_app.py`.
4. Deploy.

## Notes

- Model checkpoints are located in `models/undersampling_no_environment/`.
- These checkpoint files are large and should be tracked with Git LFS for deployment.
- The server app is in `server.py` for alternate deployment with Uvicorn.
