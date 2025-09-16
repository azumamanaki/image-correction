# -*- coding: utf-8 -*-
import os
import datetime as dt
import unicodedata
import dropbox
from dropbox.exceptions import ApiError

# === 環境変数（GitHub Secrets から渡す）===
DROPBOX_REFRESH_TOKEN = os.environ["DROPBOX_REFRESH_TOKEN"]
DROPBOX_CLIENT_ID     = os.environ["DROPBOX_CLIENT_ID"]
DROPBOX_CLIENT_SECRET = os.environ["DROPBOX_CLIENT_SECRET"]

GMAIL_FROM        = os.environ["GMAIL_FROM"]
GMAIL_TO          = os.environ["GMAIL_TO"]
GMAIL_APP_PASSWORD= os.environ["GMAIL_APP_PASSWORD"]

# === チェック対象フォルダ ===
BASES = [
    "/おうち書道/共有データ/【受講生】/【添削用　作品】/処理失敗",
    "/おうち書道/共有データ/【受講生】/【清書用　作品（出品用）】/硬筆/処理失敗",
    "/おうち書道/共有データ/【受講生】/【清書用　作品（出品用）】/毛筆/処理失敗",
    "/おうち書道/共有データ/【受講生】/【書初用　作品】/清書用/処理失敗",
    "/おうち書道/共有データ/【受講生】/【書初用　作品】/清書用/処理失敗",
]

def alt_forms(path: str):
    return {unicodedata.normalize("NFC", path), unicodedata.normalize("NFD", path)}

def get_dbx():
    return dropbox.Dropbox(
        oauth2_refresh_token=DROPBOX_REFRESH_TOKEN,
        app_key=DROPBOX_CLIENT_ID,
        app_secret=DROPBOX_CLIENT_SECRET,
        timeout=300,
    )

def count_files_recursive(dbx: dropbox.Dropbox, folder: str) -> int:
    try:
        res = dbx.files_list_folder(folder, recursive=True)
    except ApiError:
        return 0
    count = 0
    entries = res.entries
    while res.has_more:
        res = dbx.files_list_folder_continue(res.cursor)
        entries += res.entries
    for ent in entries:
        if isinstance(ent, dropbox.files.FileMetadata):
            count += 1
    return count

def build_report(dbx: dropbox.Dropbox):
    lines = []
    total = 0
    for base in BASES:
        found = 0
        for p in alt_forms(base):
            found = count_files_recursive(dbx, p)
            if found > 0:
                break
        if found > 0:  # 0件は省略
            lines.append(f"{base} に {found} 件あります。")
            total += found
    return total, "\n".join(lines)

def send_mail(subject: str, body: str):
    import smtplib
    from email.mime.text import MIMEText
    from email.utils import formatdate

    msg = MIMEText(body, _charset="utf-8")
    msg["Subject"] = subject
    msg["From"] = GMAIL_FROM
    msg["To"] = GMAIL_TO
    msg["Date"] = formatdate(localtime=True)

    with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
        smtp.starttls()
        smtp.login(GMAIL_FROM, GMAIL_APP_PASSWORD)
        smtp.send_message(msg)

def main():
    dbx = get_dbx()
    total, report = build_report(dbx)

    if total > 0:  # 1件以上ある場合のみ送信
        today = dt.datetime.utcnow().astimezone(dt.timezone(dt.timedelta(hours=9)))
        subject = f"[おうち書道] 処理失敗フォルダの検知 {today:%Y-%m-%d}（合計 {total} 件）"
        body = (
            "以下のフォルダに処理失敗データがあります。\n\n"
            f"{report}\n\n"
            "（このメールは自動送信です）"
        )
        send_mail(subject, body)
        print("Mail sent.\n" + report)
    else:
        print("No failed files. Mail not sent.")

if __name__ == "__main__":
    main()