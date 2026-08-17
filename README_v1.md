# Crypto AI Research Agent v1

Upload `crypto_ai_agent_v1.py` and `requirements.txt` to the root of your GitHub repository.

Set Streamlit Community Cloud main file path to:

```text
crypto_ai_agent_v1.py
```

Use this `requirements.txt`:

```text
streamlit
pandas
numpy
plotly
requests
scikit-learn
```

The app uses daily CoinGecko data to train a logistic-regression model and produce probability-based paper-trading signals. It does not place live trades and is not financial advice.
