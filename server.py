from __future__ import annotations

import glob
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import openpyxl
import requests
from bip_utils import (
    Bip32KeyData,
    Bip44,
    Bip44Changes,
    Bip44Coins,
    CardanoShelley,
    Cip1852,
    Cip1852Coins,
)
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


KOIOS_BASE_URL = "https://api.koios.rest/api/v1"
LOVELACE_PER_ADA = 1_000_000


def chunked(items: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def is_hex_64_bytes(value: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-fA-F]{128}", value))


def is_legacy_address(addr: str) -> bool:
    return addr.startswith("Ddz") or addr.startswith("Ae2")


def addr_prefix(addr: str) -> str:
    if addr.startswith("addr1"):
        return "addr1"
    if addr.startswith("Ddz"):
        return "Ddz"
    if addr.startswith("Ae2"):
        return "Ae2"
    if addr.startswith("stake1"):
        return "stake1"
    return "other"


def to_lovelace(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return 0
    try:
        return int((Decimal(text) * LOVELACE_PER_ADA).to_integral_value())
    except (InvalidOperation, ValueError):
        return 0


def format_ada(lovelace: int) -> str:
    ada = Decimal(lovelace) / Decimal(LOVELACE_PER_ADA)
    text = f"{ada:.6f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def parse_date_utc(text: str) -> Optional[int]:
    if not text:
        return None
    text = text.strip()
    for fmt in ("%y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        except ValueError:
            continue
    return None


def unique_preserve(items: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    ordered: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


@dataclass
class ShelleyAccount:
    xpub: str
    account_index: int = 0
    fresh_external_index: int = 0


@dataclass
class ParsedKeys:
    reward_addresses: List[str]
    shelley_accounts: List[ShelleyAccount]
    byron_xpubs: List[str]


@dataclass
class BinanceRecord:
    source: str
    tx_hash: str
    timestamp: Optional[int]
    amount_lovelace: int
    fee_lovelace: int
    address: str
    status: str
    coin: str
    network: str


class KoiosClient:
    def __init__(self, base_url: str = KOIOS_BASE_URL, timeout: int = 40) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def get(self, endpoint: str, params: Dict[str, Any]) -> Any:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        response = self.session.get(url, params=params, timeout=self.timeout)
        if response.status_code != 200:
            raise RuntimeError(
                f"Koios GET {endpoint} failed ({response.status_code}): {response.text[:280]}"
            )
        return response.json()

    def post(self, endpoint: str, payload: Dict[str, Any]) -> Any:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        response = self.session.post(url, json=payload, timeout=self.timeout)
        if response.status_code != 200:
            raise RuntimeError(
                f"Koios POST {endpoint} failed ({response.status_code}): {response.text[:280]}"
            )
        return response.json()

    def account_txs(self, stake_address: str) -> List[Dict[str, Any]]:
        return self.get("account_txs", {"_stake_address": stake_address})

    def account_addresses(self, stake_address: str, include_empty: bool = True) -> List[str]:
        payload: Dict[str, Any] = {"_stake_addresses": [stake_address]}
        if include_empty:
            payload["_empty"] = True
        rows = self.post("account_addresses", payload)
        if not rows:
            return []
        return rows[0].get("addresses", []) or []

    def address_txs(self, addresses: List[str]) -> List[Dict[str, Any]]:
        if not addresses:
            return []
        return self.post("address_txs", {"_addresses": addresses})

    def address_info(self, addresses: List[str]) -> List[Dict[str, Any]]:
        if not addresses:
            return []
        return self.post("address_info", {"_addresses": addresses})

    def tx_info(self, tx_hashes: List[str]) -> List[Dict[str, Any]]:
        if not tx_hashes:
            return []
        rows: List[Dict[str, Any]] = []
        for batch in chunked(tx_hashes, 50):
            payload = {
                "_tx_hashes": batch,
                "_inputs": True,
                "_assets": True,
                "_metadata": True,
                "_withdrawals": True,
                "_certs": True,
            }
            rows.extend(self.post("tx_info", payload))
        return rows


def _next_nonempty(lines: List[str], start_idx: int) -> str:
    for i in range(start_idx, len(lines)):
        text = lines[i].strip()
        if text:
            return text
    return ""


def _account_idx_from_label(label_line: str) -> Optional[int]:
    m = re.search(r"1815'\/(\d+)'", label_line)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _parse_fresh_path(path: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    # Expects shape like 1852'/1815'/0'/0/27
    m = re.match(r"1852'\/1815'\/(\d+)'\/(\d+)\/(\d+)$", path.strip())
    if not m:
        return None, None, None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def parse_keys_file(path: Path) -> ParsedKeys:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    rewards: List[str] = []
    byron_xpubs: List[str] = []
    shelley_map: Dict[str, ShelleyAccount] = {}

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("Byron extended public key"):
            candidate = _next_nonempty(lines, idx + 1)
            if is_hex_64_bytes(candidate):
                byron_xpubs.append(candidate.lower())
        elif stripped.startswith("Shelley extended public key"):
            candidate = _next_nonempty(lines, idx + 1)
            if is_hex_64_bytes(candidate):
                acct = _account_idx_from_label(stripped) or 0
                shelley_map[candidate.lower()] = ShelleyAccount(
                    xpub=candidate.lower(),
                    account_index=acct,
                    fresh_external_index=0,
                )
        elif stripped == "Reward address":
            candidate = _next_nonempty(lines, idx + 1)
            if candidate.startswith("stake1"):
                rewards.append(candidate)

    for match in re.finditer(r"\{[\s\S]*?\}", text):
        block = match.group(0)
        try:
            obj = json.loads(block)
        except json.JSONDecodeError:
            continue
        xpub = str(obj.get("xpub", "")).strip().lower()
        if not is_hex_64_bytes(xpub):
            continue
        fresh_path = str(obj.get("freshAddressPath", "")).strip()
        acct_idx, chain_idx, addr_idx = _parse_fresh_path(fresh_path)
        if acct_idx is None:
            continue
        existing = shelley_map.get(xpub)
        if existing is None:
            existing = ShelleyAccount(xpub=xpub, account_index=acct_idx, fresh_external_index=0)
            shelley_map[xpub] = existing
        else:
            existing.account_index = acct_idx
        if chain_idx == 0 and addr_idx is not None:
            existing.fresh_external_index = max(existing.fresh_external_index, addr_idx)

    return ParsedKeys(
        reward_addresses=unique_preserve(rewards),
        shelley_accounts=list(shelley_map.values()),
        byron_xpubs=unique_preserve(byron_xpubs),
    )


def read_xlsx_records(path: Path) -> List[Dict[str, Any]]:
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        return []
    header = [str(c).strip() if c is not None else "" for c in rows[0]]
    data: List[Dict[str, Any]] = []
    for row in rows[1:]:
        if not any(row):
            continue
        data.append(dict(zip(header, row)))
    return data


def normalize_binance_records(path: Path, source: str) -> List[BinanceRecord]:
    out: List[BinanceRecord] = []
    rows = read_xlsx_records(path)
    for row in rows:
        tx_hash = str(row.get("TXID", "") or "").strip().lower()
        if not tx_hash:
            continue
        out.append(
            BinanceRecord(
                source=source,
                tx_hash=tx_hash,
                timestamp=parse_date_utc(str(row.get("Date(UTC+0)", "") or "")),
                amount_lovelace=to_lovelace(row.get("Amount")),
                fee_lovelace=to_lovelace(row.get("Fee")),
                address=str(row.get("Address", "") or "").strip(),
                status=str(row.get("Status", "") or "").strip(),
                coin=str(row.get("Coin", "") or "").strip(),
                network=str(row.get("Network", "") or "").strip(),
            )
        )
    return out


def detect_default_file(patterns: List[str]) -> Optional[str]:
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[0]
    return None


def derive_shelley_addresses(
    account: ShelleyAccount,
    ext_limit: int,
    int_limit: int,
) -> List[str]:
    key_bytes = bytes.fromhex(account.xpub)
    key_data = Bip32KeyData(
        depth=3,
        index=(0x80000000 | account.account_index),
        chain_code=key_bytes[32:64],
        parent_fprint=b"\x00\x00\x00\x00",
    )
    bip_obj = Cip1852.FromPublicKey(
        key_bytes[0:32],
        Cip1852Coins.CARDANO_ICARUS,
        key_data=key_data,
    )
    shelley = CardanoShelley.FromCip1852Object(bip_obj)
    addresses: List[str] = []
    for idx in range(ext_limit + 1):
        addresses.append(
            shelley.Change(Bip44Changes.CHAIN_EXT).AddressIndex(idx).PublicKeys().ToAddress()
        )
    for idx in range(int_limit + 1):
        addresses.append(
            shelley.Change(Bip44Changes.CHAIN_INT).AddressIndex(idx).PublicKeys().ToAddress()
        )
    return unique_preserve(addresses)


def derive_byron_address(byron_xpub: str, chain: Bip44Changes, index: int) -> str:
    key_bytes = bytes.fromhex(byron_xpub)
    obj = Bip44.FromPublicKey(
        key_bytes[0:32],
        Bip44Coins.CARDANO_BYRON_ICARUS,
        key_data=Bip32KeyData(
            depth=3,
            index=0x80000000,
            chain_code=key_bytes[32:64],
            parent_fprint=b"\x00\x00\x00\x00",
        ),
    )
    return obj.Change(chain).AddressIndex(index).PublicKey().ToAddress()


def extract_address(entry: Dict[str, Any]) -> str:
    payment_addr = entry.get("payment_addr") or {}
    return str(payment_addr.get("bech32", "") or "")


def tx_entry_is_owned(
    entry: Dict[str, Any],
    owned_addresses: Set[str],
    reward_addresses: Set[str],
) -> bool:
    addr = extract_address(entry)
    if addr and addr in owned_addresses:
        return True
    stake_addr = entry.get("stake_addr")
    if stake_addr and stake_addr in reward_addresses:
        return True
    return False


def tx_lovelace_sum(entries: List[Dict[str, Any]]) -> int:
    return sum(int(e.get("value", "0") or 0) for e in entries)


def tx_withdrawal_sum(withdrawals: List[Dict[str, Any]], reward_addresses: Set[str]) -> int:
    total = 0
    for item in withdrawals:
        if item.get("stake_addr") in reward_addresses:
            total += int(item.get("amount", "0") or 0)
    return total


def tx_has_our_certificates(tx: Dict[str, Any], reward_addresses: Set[str]) -> bool:
    certs = tx.get("certificates") or []
    for cert in certs:
        info = cert.get("info") or {}
        for value in info.values():
            if isinstance(value, str) and value in reward_addresses:
                return True
    return False


def classify_tx(
    tx: Dict[str, Any],
    owned_addresses: Set[str],
    reward_addresses: Set[str],
    binance_deposits: Set[str],
    binance_withdrawals: Set[str],
) -> Tuple[str, List[str], int, int, int]:
    inputs = tx.get("inputs") or []
    outputs = tx.get("outputs") or []
    withdrawals = tx.get("withdrawals") or []
    certs = tx.get("certificates") or []
    fee = int(tx.get("fee", "0") or 0)

    owned_inputs = [e for e in inputs if tx_entry_is_owned(e, owned_addresses, reward_addresses)]
    owned_outputs = [e for e in outputs if tx_entry_is_owned(e, owned_addresses, reward_addresses)]

    ada_out = tx_lovelace_sum(owned_inputs)
    ada_in = tx_lovelace_sum(owned_outputs) + tx_withdrawal_sum(withdrawals, reward_addresses)
    net = ada_in - ada_out

    tags: List[str] = []
    tx_hash = tx.get("tx_hash", "")
    if tx_hash in binance_deposits:
        tags.append("binance_deposit")
    if tx_hash in binance_withdrawals:
        tags.append("binance_withdrawal")

    cert_types = {str(c.get("type", "")).lower() for c in certs}
    if any("stake_registration" in t for t in cert_types):
        tags.append("staking_in")
    if any("stake_deregistration" in t for t in cert_types):
        tags.append("staking_out")
    if any("delegation" in t for t in cert_types):
        tags.append("staking_delegation")
    if tx_withdrawal_sum(withdrawals, reward_addresses) > 0:
        tags.append("staking_reward_withdrawal")

    if ada_out > 0 and ada_in == 0:
        tags.append("transfer_out")
    elif ada_out == 0 and ada_in > 0:
        tags.append("transfer_in")
    elif ada_out > 0 and ada_in > 0:
        # This includes typical self-transfers with change.
        if abs(net + fee) <= 5_000_000:
            tags.append("self_transfer")
        else:
            tags.append("mixed_transfer")
    else:
        tags.append("other")

    priority = [
        "staking_in",
        "staking_out",
        "staking_delegation",
        "staking_reward_withdrawal",
        "binance_deposit",
        "binance_withdrawal",
        "transfer_in",
        "transfer_out",
        "self_transfer",
        "mixed_transfer",
        "other",
    ]
    primary = "other"
    for t in priority:
        if t in tags:
            primary = t
            break
    return primary, unique_preserve(tags), ada_in, ada_out, net


def collect_tx_hashes_for_addresses(
    koios: KoiosClient, addresses: Set[str], batch_size: int = 20
) -> Set[str]:
    tx_hashes: Set[str] = set()
    addr_list = sorted(addresses)
    for batch in chunked(addr_list, batch_size):
        rows = koios.address_txs(batch)
        for row in rows:
            tx_hash = str(row.get("tx_hash", "") or "").lower()
            if tx_hash:
                tx_hashes.add(tx_hash)
    return tx_hashes


def addresses_for_direct_crawl(addresses: Set[str]) -> Set[str]:
    # Account-level stake queries already cover reward-linked Shelley addresses.
    # Direct address crawl is reserved for legacy/non-stake paths.
    return {a for a in addresses if is_legacy_address(a)}


def build_reconciliation(
    keys_file: Path,
    deposit_file: Path,
    withdraw_file: Path,
    shelley_extra_gap: int,
    shelley_scan_cap: int,
    byron_scan_cap: int,
    byron_gap_limit: int,
    max_expansion_rounds: int,
    co_spend_input_cap: int,
    co_spend_output_cap: int,
    debug: bool = False,
) -> Dict[str, Any]:
    def log(msg: str) -> None:
        if debug:
            print(f"[reconcile] {msg}", flush=True)

    if not keys_file.exists():
        raise FileNotFoundError(f"Missing keys file: {keys_file}")
    if not deposit_file.exists():
        raise FileNotFoundError(f"Missing deposit report: {deposit_file}")
    if not withdraw_file.exists():
        raise FileNotFoundError(f"Missing withdraw report: {withdraw_file}")

    parsed_keys = parse_keys_file(keys_file)
    koios = KoiosClient()
    log(
        "parsed keys: rewards=%d shelley_accounts=%d byron_xpubs=%d"
        % (
            len(parsed_keys.reward_addresses),
            len(parsed_keys.shelley_accounts),
            len(parsed_keys.byron_xpubs),
        )
    )

    deposit_records = normalize_binance_records(deposit_file, "deposit")
    withdraw_records = normalize_binance_records(withdraw_file, "withdraw")
    log("parsed binance: deposits=%d withdrawals=%d" % (len(deposit_records), len(withdraw_records)))
    binance_deposit_hashes = {r.tx_hash for r in deposit_records}
    binance_withdraw_hashes = {r.tx_hash for r in withdraw_records}
    binance_deposit_addresses = {r.address for r in deposit_records if r.address}
    withdraw_destination_addresses = {r.address for r in withdraw_records if r.address}

    owned_addresses: Set[str] = set(withdraw_destination_addresses)
    reward_addresses = set(parsed_keys.reward_addresses)

    # Stake-account grounded discovery.
    all_tx_hashes: Set[str] = set(binance_deposit_hashes | binance_withdraw_hashes)
    for reward_addr in reward_addresses:
        log(f"stake crawl: account_txs/account_addresses for {reward_addr}")
        for row in koios.account_txs(reward_addr):
            tx_hash = str(row.get("tx_hash", "") or "").lower()
            if tx_hash:
                all_tx_hashes.add(tx_hash)
        for addr in koios.account_addresses(reward_addr, include_empty=True):
            if addr:
                owned_addresses.add(addr)
    log(
        "after stake crawl: tx_hashes=%d owned_addresses=%d"
        % (len(all_tx_hashes), len(owned_addresses))
    )

    # Shelley xpub grounded discovery for all key sets.
    for account in parsed_keys.shelley_accounts:
        log(f"shelley derive: account_index={account.account_index} xpub={account.xpub[:16]}...")
        ext_limit = min(
            shelley_scan_cap,
            max(account.fresh_external_index + shelley_extra_gap, shelley_extra_gap),
        )
        int_limit = min(shelley_scan_cap, max(20, account.fresh_external_index // 2 + shelley_extra_gap))
        for addr in derive_shelley_addresses(account, ext_limit, int_limit):
            owned_addresses.add(addr)
    log("after shelley derive: owned_addresses=%d" % len(owned_addresses))

    # Initial tx crawl for currently known owned addresses.
    all_tx_hashes |= collect_tx_hashes_for_addresses(
        koios, addresses_for_direct_crawl(owned_addresses)
    )
    log("after direct crawl seed: tx_hashes=%d" % len(all_tx_hashes))

    # Pull tx details, then seed legacy addresses directly from Binance deposit txs.
    tx_cache: Dict[str, Dict[str, Any]] = {}

    def ensure_tx_info(tx_ids: Set[str]) -> None:
        missing = [tx for tx in tx_ids if tx not in tx_cache]
        if not missing:
            return
        for tx in koios.tx_info(missing):
            tx_hash = str(tx.get("tx_hash", "") or "").lower()
            if tx_hash:
                tx_cache[tx_hash] = tx

    ensure_tx_info(all_tx_hashes)
    log("tx cache seeded: tx_cache=%d" % len(tx_cache))

    for record in deposit_records:
        tx = tx_cache.get(record.tx_hash)
        if not tx:
            continue
        for inp in tx.get("inputs") or []:
            addr = extract_address(inp)
            if addr and is_legacy_address(addr):
                owned_addresses.add(addr)
        for out in tx.get("outputs") or []:
            addr = extract_address(out)
            if addr and is_legacy_address(addr) and addr not in binance_deposit_addresses:
                owned_addresses.add(addr)
    log("after binance deposit legacy seeding: owned_addresses=%d" % len(owned_addresses))

    for addr in withdraw_destination_addresses:
        if is_legacy_address(addr):
            owned_addresses.add(addr)

    # Byron xpub scanning with strict gap stop, only if addresses are actually used.
    for byron_xpub in parsed_keys.byron_xpubs:
        log(f"byron scan: xpub={byron_xpub[:16]}...")
        for chain in (Bip44Changes.CHAIN_EXT, Bip44Changes.CHAIN_INT):
            consecutive_unused = 0
            idx = 0
            probe_batch = 10
            while idx < byron_scan_cap:
                stop = min(idx + probe_batch, byron_scan_cap)
                batch_addrs = [derive_byron_address(byron_xpub, chain, i) for i in range(idx, stop)]
                batch_rows = koios.address_txs(batch_addrs)
                if not batch_rows:
                    consecutive_unused += len(batch_addrs)
                    if (stop - 1) >= byron_gap_limit and consecutive_unused >= byron_gap_limit:
                        break
                    idx = stop
                    continue

                # Only fan out to single-address checks when the batch is active.
                for single_idx, addr in zip(range(idx, stop), batch_addrs):
                    rows = koios.address_txs([addr])
                    if rows:
                        consecutive_unused = 0
                        owned_addresses.add(addr)
                        for row in rows:
                            tx_hash = str(row.get("tx_hash", "") or "").lower()
                            if tx_hash:
                                all_tx_hashes.add(tx_hash)
                    else:
                        consecutive_unused += 1
                    if single_idx >= byron_gap_limit and consecutive_unused >= byron_gap_limit:
                        break
                if (stop - 1) >= byron_gap_limit and consecutive_unused >= byron_gap_limit:
                    break
                idx = stop
    log("after byron scan: tx_hashes=%d owned_addresses=%d" % (len(all_tx_hashes), len(owned_addresses)))

    # Controlled expansion, only from transactions spending owned inputs.
    for _ in range(max_expansion_rounds):
        round_idx = _ + 1
        log(f"expansion round {round_idx}: start tx_hashes={len(all_tx_hashes)} owned={len(owned_addresses)}")
        new_addresses: Set[str] = set()

        all_tx_hashes |= collect_tx_hashes_for_addresses(
            koios, addresses_for_direct_crawl(owned_addresses)
        )
        ensure_tx_info(all_tx_hashes)

        for tx_hash, tx in tx_cache.items():
            inputs = tx.get("inputs") or []
            outputs = tx.get("outputs") or []
            input_addrs = [extract_address(i) for i in inputs if extract_address(i)]
            output_addrs = [extract_address(o) for o in outputs if extract_address(o)]
            owned_input_present = any(
                tx_entry_is_owned(i, owned_addresses, reward_addresses) for i in inputs
            )
            if not owned_input_present:
                # Still pull stake-tagged addresses if present.
                for out in outputs:
                    out_addr = extract_address(out)
                    if out_addr and out.get("stake_addr") in reward_addresses:
                        new_addresses.add(out_addr)
                continue

            if len(inputs) > co_spend_input_cap or len(outputs) > co_spend_output_cap:
                continue

            for addr in input_addrs:
                new_addresses.add(addr)

            has_binance_deposit_output = any(addr in binance_deposit_addresses for addr in output_addrs)

            for out in outputs:
                out_addr = extract_address(out)
                if not out_addr or out_addr in binance_deposit_addresses:
                    continue
                if out.get("stake_addr") in reward_addresses:
                    new_addresses.add(out_addr)
                    continue
                if is_legacy_address(out_addr):
                    # Legacy mode: only infer change for explicit Binance-deposit shaped txs.
                    if has_binance_deposit_output:
                        new_addresses.add(out_addr)

            # If cert references one of our reward addresses, keep it in-scope.
            if tx_has_our_certificates(tx, reward_addresses):
                for inp in inputs:
                    addr = extract_address(inp)
                    if addr:
                        new_addresses.add(addr)

        truly_new = new_addresses - owned_addresses
        log(f"expansion round {round_idx}: newly discovered addresses={len(truly_new)}")
        if not truly_new:
            break
        owned_addresses |= truly_new

    # Final tx sync.
    all_tx_hashes |= collect_tx_hashes_for_addresses(
        koios, addresses_for_direct_crawl(owned_addresses)
    )
    ensure_tx_info(all_tx_hashes)
    log("final sync complete: tx_cache=%d owned_addresses=%d" % (len(tx_cache), len(owned_addresses)))

    records: List[Dict[str, Any]] = []
    for tx in tx_cache.values():
        tx_hash = str(tx.get("tx_hash", "") or "").lower()
        inputs = tx.get("inputs") or []
        outputs = tx.get("outputs") or []
        withdrawals = tx.get("withdrawals") or []

        primary, tags, ada_in, ada_out, net = classify_tx(
            tx=tx,
            owned_addresses=owned_addresses,
            reward_addresses=reward_addresses,
            binance_deposits=binance_deposit_hashes,
            binance_withdrawals=binance_withdraw_hashes,
        )

        owned_input_addrs = sorted(
            {
                extract_address(e)
                for e in inputs
                if tx_entry_is_owned(e, owned_addresses, reward_addresses) and extract_address(e)
            }
        )
        owned_output_addrs = sorted(
            {
                extract_address(e)
                for e in outputs
                if tx_entry_is_owned(e, owned_addresses, reward_addresses) and extract_address(e)
            }
        )
        counterparty_input_addrs = sorted(
            {
                extract_address(e)
                for e in inputs
                if not tx_entry_is_owned(e, owned_addresses, reward_addresses) and extract_address(e)
            }
        )
        counterparty_output_addrs = sorted(
            {
                extract_address(e)
                for e in outputs
                if not tx_entry_is_owned(e, owned_addresses, reward_addresses) and extract_address(e)
            }
        )
        records.append(
            {
                "tx_hash": tx_hash,
                "timestamp_unix": tx.get("tx_timestamp"),
                "timestamp_utc": datetime.fromtimestamp(
                    int(tx.get("tx_timestamp", 0) or 0), tz=timezone.utc
                ).isoformat(),
                "block_height": tx.get("block_height"),
                "type": primary,
                "tags": tags,
                "origin_wallets": sorted({extract_address(e) for e in inputs if extract_address(e)}),
                "destination_wallets": sorted(
                    {extract_address(e) for e in outputs if extract_address(e)}
                ),
                "owned_input_addresses": owned_input_addrs,
                "owned_output_addresses": owned_output_addrs,
                "counterparty_input_addresses": counterparty_input_addrs,
                "counterparty_output_addresses": counterparty_output_addrs,
                "inputs_count": len(inputs),
                "outputs_count": len(outputs),
                "ada_in_lovelace": ada_in,
                "ada_out_lovelace": ada_out,
                "fee_lovelace": int(tx.get("fee", "0") or 0),
                "net_lovelace": net,
                "ada_in": format_ada(ada_in),
                "ada_out": format_ada(ada_out),
                "fee": format_ada(int(tx.get("fee", "0") or 0)),
                "net_ada": format_ada(net),
                "withdrawal_lovelace": tx_withdrawal_sum(withdrawals, reward_addresses),
            }
        )

    records.sort(key=lambda x: (x["timestamp_unix"] or 0, x["tx_hash"]))

    owned_prefix_counts: Dict[str, int] = {"addr1": 0, "Ae2": 0, "Ddz": 0, "other": 0}
    for addr in owned_addresses:
        prefix = addr_prefix(addr)
        if prefix not in owned_prefix_counts:
            prefix = "other"
        owned_prefix_counts[prefix] += 1

    summary = {
        "transactions_total": len(records),
        "owned_addresses_total": len(owned_addresses),
        "reward_addresses_total": len(reward_addresses),
        "shelley_accounts_total": len(parsed_keys.shelley_accounts),
        "byron_xpubs_total": len(parsed_keys.byron_xpubs),
        "binance_deposit_rows": len(deposit_records),
        "binance_withdraw_rows": len(withdraw_records),
        "binance_unique_tx_hashes": len(binance_deposit_hashes | binance_withdraw_hashes),
        "earliest_tx_utc": records[0]["timestamp_utc"] if records else None,
        "latest_tx_utc": records[-1]["timestamp_utc"] if records else None,
        "owned_prefix_counts": owned_prefix_counts,
        "tx_type_counts": {},
    }
    type_counts: Dict[str, int] = {}
    for record in records:
        type_counts[record["type"]] = type_counts.get(record["type"], 0) + 1
    summary["tx_type_counts"] = type_counts

    result = {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "inputs": {
            "keys_file": str(keys_file),
            "deposit_file": str(deposit_file),
            "withdraw_file": str(withdraw_file),
        },
        "summary": summary,
        "ownership": {
            "reward_addresses": sorted(reward_addresses),
            "owned_addresses": sorted(owned_addresses),
        },
        "binance": {
            "deposit_addresses": sorted(binance_deposit_addresses),
            "deposit_tx_hashes": sorted(binance_deposit_hashes),
            "withdraw_destination_addresses": sorted(withdraw_destination_addresses),
            "withdraw_tx_hashes": sorted(binance_withdraw_hashes),
        },
        "transactions": records,
    }
    return result


class ReconcileRequest(BaseModel):
    keys_file: str = "keys"
    deposit_file: Optional[str] = None
    withdraw_file: Optional[str] = None
    shelley_extra_gap: int = Field(default=30, ge=0, le=300)
    shelley_scan_cap: int = Field(default=120, ge=10, le=500)
    byron_scan_cap: int = Field(default=80, ge=10, le=500)
    byron_gap_limit: int = Field(default=20, ge=5, le=200)
    max_expansion_rounds: int = Field(default=8, ge=1, le=30)
    co_spend_input_cap: int = Field(default=35, ge=5, le=200)
    co_spend_output_cap: int = Field(default=35, ge=5, le=200)


def resolve_inputs(req: ReconcileRequest) -> Tuple[Path, Path, Path]:
    keys_file = Path(req.keys_file)
    deposit_file = Path(
        req.deposit_file
        or detect_default_file(
            [
                "*Deposit*History*.xlsx",
                "*deposit*history*.xlsx",
                "*Deposit*.xlsx",
            ]
        )
        or ""
    )
    withdraw_file = Path(
        req.withdraw_file
        or detect_default_file(
            [
                "*Withdraw*History*.xlsx",
                "*withdraw*history*.xlsx",
                "*Withdraw*.xlsx",
            ]
        )
        or ""
    )
    return keys_file, deposit_file, withdraw_file


app = FastAPI(title="Cardano Reconciliation Crawler")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root() -> FileResponse:
    return FileResponse("static/index.html")


@app.get("/api/defaults")
def defaults() -> Dict[str, Any]:
    keys_file = "keys" if Path("keys").exists() else ""
    deposit_default = detect_default_file(
        ["*Deposit*History*.xlsx", "*deposit*history*.xlsx", "*Deposit*.xlsx"]
    )
    withdraw_default = detect_default_file(
        ["*Withdraw*History*.xlsx", "*withdraw*history*.xlsx", "*Withdraw*.xlsx"]
    )
    return {
        "keys_file": keys_file,
        "deposit_file": deposit_default or "",
        "withdraw_file": withdraw_default or "",
    }


@app.post("/api/reconcile")
def reconcile(req: ReconcileRequest) -> Dict[str, Any]:
    try:
        keys_file, deposit_file, withdraw_file = resolve_inputs(req)
        result = build_reconciliation(
            keys_file=keys_file,
            deposit_file=deposit_file,
            withdraw_file=withdraw_file,
            shelley_extra_gap=req.shelley_extra_gap,
            shelley_scan_cap=req.shelley_scan_cap,
            byron_scan_cap=req.byron_scan_cap,
            byron_gap_limit=req.byron_gap_limit,
            max_expansion_rounds=req.max_expansion_rounds,
            co_spend_input_cap=req.co_spend_input_cap,
            co_spend_output_cap=req.co_spend_output_cap,
        )
        out_dir = Path("output")
        out_dir.mkdir(parents=True, exist_ok=True)
        latest_path = out_dir / "latest-reconciliation.json"
        latest_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
