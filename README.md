1.  To run code:

```python -m streamlit run app.py```
or 
```C:\Python313\python.exe -m streamlit run app.py```

2. Install for the API
```pip install streamlit pandas google-genai plotly```

3. Create your own API key from google gemini
~Go to: ```https://aistudio.google.com/``` 
~Make your API key
~go to the ```.streamlit``` folder
~inside this folder, make a ```secrets.toml``` file
~write in the file ```GEMINI_API_KEY = "ENTER YOUR API KEY HERE"```
~now you can run it