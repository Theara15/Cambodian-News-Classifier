# Deploy this Streamlit app

## 1. Install Python dependencies

From the `f:\app` folder:

```powershell
python -m pip install -r requirements.txt
```

## 2. Run locally

```powershell
cd f:\app
streamlit run streamlit_app.py
```

## 3. Deploy to Streamlit Cloud

1. Push the `f:\app` folder to a GitHub repository.
2. In Streamlit Cloud, create a new app and connect your GitHub repo.
3. Set the app entrypoint to:

```text
streamlit_app.py
```

4. Use the default branch or choose the branch you pushed.

## 4. Notes

- Model checkpoints are under `models/undersampling_no_environment/`.
- Streamlit Cloud deployments must include those `.pt` files or track them with Git LFS.
- If Streamlit Cloud fails because the repo is not in the project root, make sure the app folder is the repository root or set the app path correctly.
- If you want a simpler web service instead, use `python server.py` and deploy with a platform that supports `uvicorn`.
