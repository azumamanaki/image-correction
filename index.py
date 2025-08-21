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
DROPBOX_DEBUG_FOLDER = DROPBOX_SRC_FOLDER + "/_debug"

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

# ===== PDF → 画像変換 =====
def pdf_to_images(pdf_bytes):
    images = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for i, page in enumerate(doc):
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
        print(f"  [pdf] {i+1} ページ目を画像化 (size={pix.width}x{pix.height})")
    return images

# ===== 白背景除去による半紙抽出 =====
def extract_hanshi_white_bg(image, output_size=(2480, 3508), sat_thresh=30, val_thresh=200, debug_save=True):
    img = image.convert("RGB")
    np_img = np.array(img)

    # HSV変換
    hsv = cv2.cvtColor(np_img, cv2.COLOR_RGB2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # 白背景以外のマスク
    mask = (s > sat_thresh) | (v < val_thresh)
    mask = mask.astype(np.uint8) * 255

    # 輪郭検出
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("  [hanshi_white] 抽出対象なし → 元画像使用")
        return img

    # 最大輪郭を採用
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    crop_img = img.crop((x, y, x+w, y+h))

    # A4サイズにリサイズ
    crop_img = crop_img.resize(output_size, Image.LANCZOS)

    # デバッグ保存
    if debug_save:
        debug_img = np_img.copy()
        cv2.drawContours(debug_img, [c], -1, (255,0,0), 3)
        debug_pil = Image.fromarray(debug_img)
        debug_bytes = io.BytesIO()
        debug_pil.save(debug_bytes, format="PNG")
        debug_bytes.seek(0)
        debug_path = f"{DROPBOX_DEBUG_FOLDER}/hanshi_debug.png"
        try:
            dbx.files_upload(debug_bytes.read(), debug_path, mode=dropbox.files.WriteMode("overwrite"))
            print(f"  [DEBUG] トリミング結果を {debug_path} に保存")
        except Exception as e:
            print(f"  [DEBUG_ERROR] 保存失敗: {e}")

    print("  [hanshi_white] 白背景除去・トリミング・A4変換成功")
    return crop_img

# ===== Dropbox ファイル処理 =====
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

            # 初期縦横判定（横長なら回転）
            w, h = img.size
            if w > h:
                img = img.rotate(90, expand=True)
                print(f"  [INIT ROTATE] 横長画像 → 90°回転")
            else:
                print(f"  [INIT ROTATE] 縦長画像 → 回転なし")

            # 白背景除去による半紙抽出
            img = extract_hanshi_white_bg(img, output_size=(2480, 3508), debug_save=True)

            processed_images.append(img)

        # PDF 保存
        pdf_bytes = io.BytesIO()
        processed_images[0].save(pdf_bytes, format="PDF", save_all=True, append_images=processed_images[1:])
        pdf_bytes.seek(0)
        dest_pdf_path = f"{DROPBOX_PRINT_FOLDER}/{os.path.splitext(file_name)[0]}.pdf"
        dbx.files_upload(pdf_bytes.read(), dest_pdf_path, mode=dropbox.files.WriteMode("overwrite"))
        print(f"[UPLOAD] {dest_pdf_path} にアップロード完了")

        # 元ファイル移動
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