# -*- coding: utf-8 -*-
"""
create_folder.py
- 今月の "YYYY-M" フォルダを指定フォルダ群の直下と、
  顧客名簿/顧客ファイル/xxxxx/ の各サブフォルダ直下に作成する。
- 既に存在していればスキップ（冪等）。
"""

import os
from datetime import date
import dropbox
from dropbox.exceptions import ApiError
from dropbox.files import WriteMode

# ==== Dropbox クライアント ====
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

# ==== 共通 ====
def ym_label(d: date) -> str:
    # 例: 2025-9（先頭ゼロ無し）
    return f"{d.year}-{d.month}"

def ensure_folder(dbx: dropbox.Dropbox, path: str) -> None:
    """
    Dropbox はパスの全段が存在している必要がある。
    files_create_folder_v2 は既存ならエラーだが、already_exists は無視する。
    """
    if path == "/" or path == "":
        return
    # 先に親を作る
    parent = "/".join(path.rstrip("/").split("/")[:-1])
    if parent and parent != "":
        ensure_folder(dbx, parent)
    try:
        dbx.files_create_folder_v2(path)
        print(f"[create] {path}")
    except ApiError as e:
        # 既にある場合は無視
        if getattr(e, "error", None) and e.error.is_path() and e.error.get_path().is_conflict():
            # 競合=既存
            print(f"[exists] {path}")
        else:
            raise

def list_subfolders(dbx: dropbox.Dropbox, base: str):
    """直下のフォルダ一覧（FolderMetadata のみ）"""
    res = dbx.files_list_folder(base)
    entries = res.entries
    while res.has_more:
        res = dbx.files_list_folder_continue(res.cursor)
        entries.extend(res.entries)
    for ent in entries:
        if ent.__class__.__name__ == "FolderMetadata":
            yield ent.name

def main():
    dbx = get_dbx()
    today = date.today()
    ym = ym_label(today)

    # 対象の固定フォルダ（6 か所）
    FIXED_BASES = [
        "/おうち書道/共有データ/【受講生】/【添削用　作品】/補正済元画像",
        "/おうち書道/共有データ/【受講生】/【添削用　作品】/添削用印刷済",
        "/おうち書道/共有データ/【受講生】/【清書用　作品（出品用）】/毛筆/補正済元画像",
        "/おうち書道/共有データ/【受講生】/【清書用　作品（出品用）】/毛筆/清書用印刷済",
        "/おうち書道/共有データ/【受講生】/【清書用　作品（出品用）】/硬筆/補正済元画像",
        "/おうち書道/共有データ/【受講生】/【清書用　作品（出品用）】/硬筆/清書用印刷済",
    ]

    # 顧客ファイルのルート
    CUSTOMERS_ROOT = "/おうち書道/共有データ/顧客名簿/顧客ファイル"

    # 1) 固定フォルダ直下に YYYY-M
    for base in FIXED_BASES:
        ensure_folder(dbx, f"{base}/{ym}")

    # 2) 顧客ファイル/xxxxx 直下に YYYY-M
    for sub in list_subfolders(dbx, CUSTOMERS_ROOT):
        ensure_folder(dbx, f"{CUSTOMERS_ROOT}/{sub}/{ym}")

    print("[done] create_folder.py 完了")

if __name__ == "__main__":
    main()