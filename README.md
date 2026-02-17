# Cardano Reconciliation Crawler

Local web app to reconcile Cardano transaction history by combining:

- `keys` (all reward addresses, Shelley xpubs, Byron xpubs found in file)
- Binance deposit export (`.xlsx`)
- Binance withdraw export (`.xlsx`)
- On-chain data from Koios (`https://api.koios.rest`)

## Run

```bash
python -m pip install -r requirements.txt
python -m uvicorn server:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`.

## Output

- Web dashboard summary and transaction table
- Full JSON export at `output/latest-reconciliation.json`

## Notes

- Crawler is intentionally scoped: it expands ownership from transactions that spend known-owned inputs, with complexity caps to avoid exchange rabbit holes.
- Binance deposit addresses are treated as external.
- All key sets in `keys` are parsed and used.
