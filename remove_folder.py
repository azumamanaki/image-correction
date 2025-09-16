# -*- coding: utf-8 -*-
"""
remove_folder.py
- 実行月より3ヶ月前（それ以前）の "YYYY-M" または "YYYY-MM" という名前の
  月フォルダを指定フォルダ群と顧客ファイル配下から削除する。
  例: 2025-09 実行なら 2025-06 以前を削除。
"""

import os
from datetime import date
import dropbox
from dropbox.exceptions import ApiError
import re

def get_dbx():
    refresh = os.environ.get("DROPBOX_REFRESH_TOKEN")
    app_key = os.environ.get("DROPBOX_CLIENT_ID")
    app_secret = os.environ.get("DROPBOX_CLIENT_SECRET")
    if not (refresh and app_key and app_secret):
        raise RuntimeError("Dropbox 認証情報が環境変数にありません。")
    return dropbox.Dropbox(
        oauth2_refresh_token=refresh,
        app_key=app_key,
        app_secret=app_secret,
    )

def list_entries(dbx: dropbox.Dropbox, base: str):
    res = dbx.files_list_folder(base)
    entries = res.entries
    while res.has_more:
        res = dbx.files_list_folder_continue(res.cursor)
        entries.extend(res.entries)
    return entries

def list_subfolders(dbx: dropbox.Dropbox, base: str):
    for ent in list_entries(dbx, base):
        if ent.__class__.__name__ == "FolderMetadata":
            yield ent

def parse_ym(name: str):
    """
    'YYYY-M' or 'YYYY-MM' を (year, month) に。
    不一致は None。
    """
    m = re.fullmatch(r"(\d{4})-(\d{1,2})", name)
    if not m:
        return None
    y = int(m.group(1))
    mo = int(m.group(2))
    if 1 <= mo <= 12:
        return (y, mo)
    return None

def ym_less(a, b):
    """(y,m) の辞書式比較"""
    ay, am = a
    by, bm = b
    return (ay < by) or (ay == by and am < bm)

def minus_months(y: int, m: int, k: int):
    """(y,m) から k ヶ月戻る"""
    idx = (y * 12 + (m - 1)) - k
    return (idx // 12, (idx % 12) + 1)

def cutoff_ym(today: date):
    """
    3ヶ月前の (year, month) を返す。
    '以前' を削除するので、判定は < cutoff ではなく <= cutoff。
    例: 2025-09 → cutoff = (2025, 6)
    """
    y, m = today.year, today.month
    return minus_months(y, m, 3)

def safe_delete_folder(dbx: dropbox.Dropbox, path: str):
    try:
        dbx.files_delete_v2(path)
        print(f"[delete] {path}")
    except ApiError as e:
        print(f"[skip] delete error: {path} -> {e}")

def main():
    dbx = get_dbx()
    today = date.today()
    cutoff = cutoff_ym(today)  # これ以前を削除（<=）

    FIXED_BASES = [
        "/おうち書道/共有データ/【受講生】/【添削用　作品】/補正済元画像",
        "/おうち書道/共有データ/【受講生】/【添削用　作品】/添削用印刷済",
        "/おうち書道/共有データ/【受講生】/【清書用　作品（出品用）】/毛筆/補正済元画像",
        "/おうち書道/共有データ/【受講生】/【清書用　作品（出品用）】/毛筆/清書用印刷済",
        "/おうち書道/共有データ/【受講生】/【清書用　作品（出品用）】/硬筆/補正済元画像",
        "/おうち書道/共有データ/【受講生】/【清書用　作品（出品用）】/硬筆/清書用印刷済",
        "/おうち書道/共有データ/【受講生】/【書初用　作品】/清書用/補正済元画像",
        "/おうち書道/共有データ/【受講生】/【書初用　作品】/清書用/清書用印刷済",
    ]
    CUSTOMERS_ROOT = "/おうち書道/共有データ/顧客名簿/顧客ファイル"

    # 1) 固定フォルダ配下の YYYY-M* を削除
    for base in FIXED_BASES:
        for sub in list_subfolders(dbx, base):
            parsed = parse_ym(sub.name)
            if parsed and (parsed == cutoff or ym_less(parsed, cutoff)):
                safe_delete_folder(dbx, f"{base}/{sub.name}")

    # 2) 顧客ファイル/xxxxx 配下の YYYY-M* を削除
    for cust in list_subfolders(dbx, CUSTOMERS_ROOT):
        cust_path = f"{CUSTOMERS_ROOT}/{cust.name}"
        for sub in list_subfolders(dbx, cust_path):
            parsed = parse_ym(sub.name)
            if parsed and (parsed == cutoff or ym_less(parsed, cutoff)):
                safe_delete_folder(dbx, f"{cust_path}/{sub.name}")

    print("[done] remove_folder.py 完了")

if __name__ == "__main__":
    main()