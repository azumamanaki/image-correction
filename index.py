import dropbox
import os
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import datetime

# Dropbox アクセストークン
ACCESS_TOKEN = "<あなたのリフレッシュトークンで取得したアクセストークン>"
dbx = dropbox.Dropbox(ACCESS_TOKEN)

# フォルダパス
SRC_FOLDER = "/おうち書道/共有データ/【受講生】/【添削用　作品】/添削用印刷未"
DST_FOLDER = "/おうち書道/共有データ/【受講生】/【添削用　作品】/補正済元画像"
ERR_FOLDER = "/おうち書道/共有データ/【受講生】/【添削用　作品】/補正失敗"

LOG_FILE = "process_log.txt"

# ログ出力
def write_log(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")

# 画像補正処理
def correct_image(image_bytes):
    try:
        # OpenCVで読み込み
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150, apertureSize=3)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) == 4:
                pts = approx.reshape(4, 2)
                rect = np.zeros((4, 2), dtype="float32")
                s = pts.sum(axis=1)
                rect[0] = pts[np.argmin(s)]
                rect[2] = pts[np.argmax(s)]
                diff = np.diff(pts, axis=1)
                rect[1] = pts[np.argmin(diff)]
                rect[3] = pts[np.argmax(diff)]
                (tl, tr, br, bl) = rect
                width = 2480  # A4幅(px, 300dpi)
                height = 3508 # A4高さ(px, 300dpi)
                dst_pts = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype="float32")
                M = cv2.getPerspectiveTransform(rect, dst_pts)
                warp = cv2.warpPerspective(img, M, (width, height))
                _, buf = cv2.imencode(".jpg", warp)
                return buf.tobytes()
    except Exception as e:
        write_log(f"補正処理中にエラー: {str(e)}")
    return None

# メイン処理
def process_files():
    try:
        res = dbx.files_list_folder(SRC_FOLDER)
        for entry in res.entries:
            if isinstance(entry, dropbox.files.FileMetadata):
                filename = entry.name
                ext = os.path.splitext(filename)[1].lower()

                write_log(f"処理開始: {filename}")

                if ext not in [".png", ".jpg", ".jpeg", ".pdf"]:
                    dbx.files_move_v2(entry.path_lower, f"{ERR_FOLDER}/{filename}", allow_shared_folder=True, autorename=True)
                    write_log(f"未対応拡張子 → 補正失敗フォルダ移動: {filename}")
                    continue

                _, res_file = dbx.files_download(entry.path_lower)
                image_bytes = res_file.content

                corrected = correct_image(image_bytes)
                if corrected:
                    dst_path = f"{DST_FOLDER}/{filename}"
                    dbx.files_upload(corrected, dst_path, mode=dropbox.files.WriteMode("overwrite"))
                    write_log(f"補正済元画像に保存完了: {filename}")

                    # 添削用印刷未にもコピー
                    dst_copy = f"{SRC_FOLDER}/{filename}"
                    dbx.files_upload(corrected, dst_copy, mode=dropbox.files.WriteMode("overwrite"))
                    write_log(f"添削用印刷未にコピー完了: {filename}")

                    # 元ファイル削除
                    dbx.files_delete_v2(entry.path_lower)
                    write_log(f"削除完了: {filename}")
                else:
                    dbx.files_move_v2(entry.path_lower, f"{ERR_FOLDER}/{filename}", allow_shared_folder=True, autorename=True)
                    write_log(f"補正失敗フォルダに移動: {filename}")

    except Exception as e:
        write_log(f"処理全体でエラー: {str(e)}")

if __name__ == "__main__":
    process_files()
    print("処理完了しました。ログは process_log.txt を確認してください。")