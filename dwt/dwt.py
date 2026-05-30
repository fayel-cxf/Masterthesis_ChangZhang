"""
Equasis DWT Scraper
====================
Batch query vessel DWT information from Equasis website
Please register a free account at https://www.equasis.org before use

Usage:
    python equasis_dwt_scraper.py

Configuration (modify the username, password, and file paths below):
"""

from unittest import result

import requests
import pandas as pd
import time
import random
import logging
import os
from bs4 import BeautifulSoup

# ============================================================
# ✏️  Please modify here
# ============================================================
#EQUASIS_EMAIL    = ""   # Your Equasis registered email
#EQUASIS_PASSWORD = ""             # Your Equasis password
#EQUASIS_EMAIL    = ""   # Your Equasis registered email
#EQUASIS_PASSWORD = ""             # Your Equasis password
#EQUASIS_EMAIL    = ""   # Your Equasis registered email
#EQUASIS_PASSWORD = ""             # Your Equasis password
BASE_DIR   = Path(__file__).resolve().parent.parent.parent
INPUT_CSV    = BASE_DIR / "data/raw/equasis_list_B_nbic_sample.csv"#INPUT_CSV ="equasis_list_A_voyage_vessels"
OUTPUT_CSV   = BASE_DIR / "data/raw/list_B_with_dwt.csv"
PROGRESS_CSV = BASE_DIR / "data/raw/list_B_progress.csv"
# ============================================================

# Request interval (seconds): Random between MIN ~ MAX to avoid IP ban
DELAY_MIN = 3.0
DELAY_MAX = 7.0

# Pause longer after how many consecutive failures
PAUSE_AFTER_FAILURES = 10
LONG_PAUSE_SECONDS   = 120

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("scraper.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Equasis Session Management
# ──────────────────────────────────────────────

def create_session() -> requests.Session:
    """Create session with browser User-Agent"""
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-GB,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Referer": "https://www.equasis.org/",
    })
    return s


def login(session: requests.Session) -> bool:
    """Log in to Equasis, return True if successful"""
    login_url = "https://www.equasis.org/EquasisWeb/authen/HomePage?fs=HomePage"
    post_url  = "https://www.equasis.org/EquasisWeb/authen/HomePage?fs=HomePage"

    # GET homepage first to fetch hidden token (if any)
    try:
        resp = session.get(login_url, timeout=20)
        resp.raise_for_status()
    except Exception as e:
        log.error(f"Unable to access Equasis homepage: {e}")
        return False

    soup = BeautifulSoup(resp.text, "html.parser")

    # Construct login form payload
    payload = {
        "j_email":    EQUASIS_EMAIL,
        "j_password": EQUASIS_PASSWORD,
        "submit":     "Login",
    }

    # Some page versions contain hidden fields, include them all
    for hidden in soup.select("form input[type=hidden]"):
        name = hidden.get("name")
        val  = hidden.get("value", "")
        if name:
            payload[name] = val

    try:
        resp = session.post(post_url, data=payload, timeout=20)
        resp.raise_for_status()
    except Exception as e:
        log.error(f"Login request failed: {e}")
        return False

    # Check if login is successful (no "Login" text / username appears in the page)
    if "j_password" in resp.text or "Invalid" in resp.text:
        log.error("Login failed, please check your email and password")
        return False

    log.info("✅ Equasis login successful")
    return True


# ──────────────────────────────────────────────
# Single IMO Query
# ──────────────────────────────────────────────

def query_dwt(session: requests.Session, imo: str) -> dict:
    """
    Query vessel information for a single IMO.
    Returns a dict containing fields like dwt, gross_tonnage, ship_type, flag, etc.
    Corresponding fields will be None if the query fails.
    """
    result = {
        "IMO":           imo,
        "DWT":           None,
        "Gross_Tonnage": None,
        "Ship_Type":     None,
        "Flag":          None,
        "Status":        "pending",
    }

    url = "https://www.equasis.org/EquasisWeb/restricted/ShipInfo"
    params = {"fs": "Search", "P_IMO": imo}

    try:
        resp = session.get(url, params=params, timeout=25)
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        result["Status"] = "timeout"
        log.warning(f"IMO {imo}: Request timeout")
        return result
    except Exception as e:
        result["Status"] = f"error: {e}"
        log.warning(f"IMO {imo}: Request error {e}")
        return result

    if "j_password" in resp.text:
        result["Status"] = "session_expired"
        log.warning(f"IMO {imo}: Session expired, re-login required")
        return result

    soup = BeautifulSoup(resp.text, "html.parser")

    # ── Parse DWT ──
    # Equasis page structure: inside the td next to the row containing "Deadweight"
    dwt = _extract_field(soup, ["DWT"])
    if dwt:
        # Remove commas, units, etc., and keep only digits
        dwt_clean = "".join(c for c in dwt if c.isdigit())
        result["DWT"] = int(dwt_clean) if dwt_clean else None

    result["Gross_Tonnage"] = _extract_field(soup, ["Gross tonnage"])
    result["Ship_Type"]     = _extract_field(soup, ["Type of ship"])
    result["Flag"]          = _extract_field(soup, ["Flag"])
    
    if result["DWT"] is not None:
        result["Status"] = "ok"
        log.info(f"IMO {imo}: DWT = {result['DWT']}")
    else:
        result["Status"] = "not_found"
        log.warning(f"IMO {imo}: DWT not found (field may not exist on the page)")

    return result


def _extract_field(soup: BeautifulSoup, keywords: list):
    # Find all td tags containing keywords, get the td value next to it or in the same row
    for tag in soup.find_all(["td", "th", "div", "span", "p"]):
        text = tag.get_text(strip=True)
        for kw in keywords:
            if kw.lower() == text.lower():
                # Find the next sibling at the same level
                sibling = tag.find_next_sibling()
                if sibling:
                    val = sibling.get_text(strip=True)
                    if val and val not in ("-", "N/A", ""):
                        return val
                # Find the next td under the parent
                parent = tag.parent
                if parent:
                    tds = parent.find_all(["td", "span"])
                    for i, td in enumerate(tds):
                        if kw.lower() in td.get_text(strip=True).lower():
                            if i + 1 < len(tds):
                                val = tds[i+1].get_text(strip=True)
                                if val and val not in ("-", "N/A", ""):
                                    return val
    return None

# ──────────────────────────────────────────────
# Main Process
# ──────────────────────────────────────────────

def load_progress() -> set:
    """Read the set of completed IMOs (for breakpoint resume)"""
    if os.path.exists(PROGRESS_CSV):
        df = pd.read_csv(PROGRESS_CSV, dtype=str)
        done = set(df[df["Status"] == "ok"]["IMO"].tolist())
        log.info(f"Breakpoint resume: {len(done)} items completed")
        return done
    return set()


def main():
    # Read input
    df_input = pd.read_csv(INPUT_CSV, dtype=str)
    imos = df_input["imo"].dropna().unique().tolist()
    log.info(f"Total {len(imos)} IMOs to be queried")

    # Breakpoint resume: skip successfully completed ones
    done_imos = load_progress()
    imos_todo = [i for i in imos if i not in done_imos]
    log.info(f"Remaining {len(imos_todo)} items after removing completed ones")

    # Login
    session = create_session()
    if not login(session):
        log.error("Login failed, exiting")
        return

    results = []
    # Read existing progress
    if os.path.exists(PROGRESS_CSV):
        results = pd.read_csv(PROGRESS_CSV, dtype=str).to_dict("records")

    consecutive_failures = 0

    for idx, imo in enumerate(imos_todo, 1):
        log.info(f"Progress: {idx}/{len(imos_todo)}  IMO: {imo}")

        result = query_dwt(session, imo)

        # Session expired → Re-login
        if result["Status"] == "session_expired":
            log.info("Attempting to re-login...")
            session = create_session()
            if login(session):
                result = query_dwt(session, imo)
            else:
                log.error("Re-login failed, exiting")
                break

        results.append(result)

        # Record consecutive failure count
        if result["Status"] in ("ok", "not_found"):
            consecutive_failures = 0
        else:
            consecutive_failures += 1

        # Save progress every 10 items
        if idx % 10 == 0:
            pd.DataFrame(results).to_csv(PROGRESS_CSV, index=False)
            log.info(f"Progress saved ({idx} items)")

        # Long pause if too many consecutive failures
        if consecutive_failures >= PAUSE_AFTER_FAILURES:
            log.warning(f"Consecutive failures reached {consecutive_failures} times, pausing for {LONG_PAUSE_SECONDS}s ...")
            time.sleep(LONG_PAUSE_SECONDS)
            consecutive_failures = 0
            # Re-login
            session = create_session()
            login(session)

        # Normal request interval
        delay = random.uniform(DELAY_MIN, DELAY_MAX)
        time.sleep(delay)

    # Final save
    df_results = pd.DataFrame(results)
    pd.DataFrame(results).to_csv(PROGRESS_CSV, index=False)

    # Merge back to original CSV
    df_output = df_input.merge(
        df_results[["IMO", "DWT", "Gross_Tonnage", "Ship_Type", "Flag", "Status"]],
        on="IMO",
        how="left"
    )
    df_output.to_csv(OUTPUT_CSV, index=False)

    # Statistics
    ok_count  = (df_results["Status"] == "ok").sum()
    nf_count  = (df_results["Status"] == "not_found").sum()
    err_count = len(df_results) - ok_count - nf_count
    log.info("=" * 50)
    log.info(f"Done! Success: {ok_count}  Not Found: {nf_count}  Error: {err_count}")
    log.info(f"Results saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()