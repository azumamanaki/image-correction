import os
import io
import dropbox
import requests
import numpy as np
from PIL import Image, ImageOps
import cv2
import fitz  # PyMuPDF

# ===== 設定 =====
DROPBOX_CLIENT_ID = os.environ["DROPBOX_CLIENT_ID"]
DROPBOX_CLIENT_SECRET = os.environ["DROPBOX_CLIENT_SECRET"]
DROPBOX_REFRESH_TOKEN = os.environ["DROPBOX_REFRESH_TOKEN"]

DROPBOX_SRC_FOLDER = "/おうち書道/共有データ/【受講生】/【添削用　作品】"
DROPBOX_PRINT_FOLDER = DROPBOX_SRC_FOLDER + "/添削用印刷未"
DROPBOX_PROCESSED_FOLDER = DROPBOX_SRC_FOLDER + "/補正済元画像"
DROPBOX_FAILED_FOLDER = DROPBOX_SRC_FOLDER + "/補正失敗"

SUPPORTED_EXTS = [".png", ".jpeg", ".jpg", ".pdf"]

# ===== 初期化 =====
def get_access_token():
    data = {
        "grant_type": "refresh_token",
        "refresh_token": DROPBOX_REFRESH_TOKEN,
        "client_id": DROPBOX_CLIENT_ID,
        "client_secret": DROPBOX_CLIENT_SECRET
    }
    resp = requests.post("https://api.dropbox.com/oauth2/token", data=data)
    resp.raise_for_status()
    return resp.json()["access_token"]

dbx = dropbox.Dropbox(get_access_token())

# ===== 半紙検出 =====
def extract_hanshi(image, output_size=(2480, 3508)):
    """書道の半紙を検出し透視変換で正面化する"""
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    hanshi_contour = None
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            hanshi_contour = approx
            break

    if hanshi_contour is None:
        print("  [hanshi] 半紙検出失敗 → 元画像を使用")
        return image

    # 頂点を (左上,右上,右下,左下) に並べ替え
    pts = hanshi_contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # 左上
    rect[2] = pts[np.argmax(s)]   # 右下

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 右上
    rect[3] = pts[np.argmax(diff)]  # 左下

    dst = np.array([
        [0, 0],
        [output_size[0]-1, 0],
        [output_size[0]-1, output_size[1]-1],
        [0, output_size[1]-1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(img_cv, M, output_size)

    print("  [hanshi] 半紙検出・透視変換成功")
    return Image.fromarray(cv2.cvtColor(warp, cv2.COLOR_BGR2RGB))

# ===== Dropboxファイル処理 =====
def pdf_to_images(pdf_bytes):
    images = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for i, page in enumerate(doc):
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
        print(f"  [pdf] {i+1} ページ目を画像化 (size={pix.width}x{pix.height})")
    return images

def process_file(file_metadata):
    file_path = file_metadata.path_lower
    file_name = os.path.basename(file_path)
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    print(f"\n=== 処理開始: {file_name} (拡張子={ext}) ===")

    if ext not in SUPPORTED_EXTS:
        fail_dest = f"{DROPBOX_FAILED_FOLDER}/{file_name}"
        try:
            dbx.files_move_v2(file_path, fail_dest, autorename=True)
            print(f"[SKIP] 非対応拡張子 → {fail_dest} に移動")
        except Exception as e:
            print(f"[ERROR] 非対応ファイルの移動失敗 {file_path}: {e}")
        return

    try:
        print("  [DL] ダウンロード中...")
        _, res = dbx.files_download(file_path)
        data = res.content
        print(f"  [DL] ダウンロード完了 ({len(data)} bytes)")

        if ext == ".pdf":
            print("  [LOAD] PDF → 画像変換開始")
            images = pdf_to_images(data)
        else:
            print("  [LOAD] 画像読み込み中...")
            img = Image.open(io.BytesIO(data))
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
            print(f"  [LOAD] 読み込み完了 size={img.width}x{img.height}")
            images = [img]

        processed_images = []
        for idx, img in enumerate(images):
            print(f"  [PROC] {idx+1}枚目 補正開始")

            img = extract_hanshi(img, output_size=(2480, 3508))

            w, h = img.size
            if w > h:
                img = img.rotate(90, expand=True)
                print(f"  [ROTATE] 横長検出 → 90°回転")

            img = img.resize((2480, 3508), Image.LANCZOS)
            print(f"  [PROC] A4サイズにリサイズ ({img.width}x{img.height})")

            processed_images.append(img)

        pdf_bytes = io.BytesIO()
        processed_images[0].save(pdf_bytes, format="PDF", save_all=True, append_images=processed_images[1:])
        pdf_bytes.seek(0)

        dest_pdf_path = f"{DROPBOX_PRINT_FOLDER}/{os.path.splitext(file_name)[0]}.pdf"
        dbx.files_upload(pdf_bytes.read(), dest_pdf_path, mode=dropbox.files.WriteMode("overwrite"))
        print(f"[UPLOAD] {dest_pdf_path} にアップロード完了")

        processed_dest = f"{DROPBOX_PROCESSED_FOLDER}/{file_name}"
        dbx.files_move_v2(file_path, processed_dest, autorename=True)
        print(f"[MOVE] 元ファイルを {processed_dest} に移動完了")

        print(f"✅ 成功: {file_name}")

    except Exception as e:
        fail_dest = f"{DROPBOX_FAILED_FOLDER}/{file_name}"
        try:
            dbx.files_move_v2(file_path, fail_dest, autorename=True)
            print(f"❌ 失敗: {file_name} → {fail_dest}")
        except Exception as move_err:
            print(f"[ERROR] 失敗ファイルの移動失敗 {file_path}: {move_err}")
        print(f"[ERROR] {file_name} 処理中に例外発生: {e}")

# ===== メイン =====
def main():
    print("=== Dropbox フォルダ走査開始 ===")
    res = dbx.files_list_folder(DROPBOX_SRC_FOLDER)
    for entry in res.entries:
        if isinstance(entry, dropbox.files.FileMetadata):
            process_file(entry)

if __name__ == "__main__":
    main()