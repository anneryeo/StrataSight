this folder is for initial tests and updates 

streamlit run src\stratasight.py --server.port 8501
Get-Content .\streamlit.log -Wait -Tail 200

streamlit run src\stratasight.py --server.port 8501 --logger.level debug 2>&1 | Tee-Object -FilePath .\streamlit.log -Append

streamlit run src\stratasight.py --server.port 8501 --logger.level debug > .\streamlit.log 2>&1

Start-Job -ScriptBlock {
  . "$PWD\.venv\Scripts\Activate.ps1"
  streamlit run src\stratasight.py --server.port 8501 --logger.level debug 2>&1 | Tee-Object -FilePath "$PWD\streamlit.log" -Append
} -Name StreamlitJob

pip install -r requirements.txt