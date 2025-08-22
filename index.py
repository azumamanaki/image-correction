# -*- coding: utf-8 -*-
import os
import io
import dropbox
import requests
import numpy as np
from PIL import Image, ImageOps
import cv2
import fitz  # PyMuPDF

# ===== 設定（必須環境変数） =====
DROPBOX_CLIENT_ID = os.environ["DROPBOX_CLIENT_ID"]
DROPBOX_CLIENT_SECRET = os.environ["DROPBOX_CLIENT_SECRET"]
DROPBOX_REFRESH_TOKEN = os.environ["DROPBOX_REFRESH_TOKEN"]

DROPBOX_SRC_FOLDER = "/おうち書道/共有データ/【受講生】/【添削用　作品】"
DROPBOX_PRINT_FOLDER = DROPBOX_SRC_FOLDER + "/添削用印刷未"
DROPBOX_PROCESSED_FOLDER = DROPBOX_SRC_FOLDER + "/補正済元画像"
DROPBOX_FAILED_FOLDER = DROPBOX_SRC_FOLDER + "/補正失敗"
DROPBOX_DEBUG_FOLDER = DROPBOX_SRC_FOLDER + "/_debug"

SUPPORTED_EXTS = [".png", ".jpeg", ".jpg", ".pdf"]

DEBUG_SAVE = True  # debug画像をDropbox/_debug に保存

# ===== Dropbox 初期化 =====
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

# ===== デバッグ保存 =====
def save_debug_to_dropbox(pil_img, name):
    if not DEBUG_SAVE:
        return
    try:
        bio = io.BytesIO()
        pil_img.save(bio, format="PNG")
        bio.seek(0)
        path = f"{DROPBOX_DEBUG_FOLDER}/{name}"
        dbx.files_upload(bio.read(), path, mode=dropbox.files.WriteMode("overwrite"))
        print(f"  [DEBUG_SAVE] {path} に保存")
    except Exception as e:
        print(f"  [DEBUG_SAVE_ERROR] {e}")

# ===== 高精度トリミング =====
def trim_paper_cv(pil_img, output_width=2400, debug=True, file_name="img", page_idx=0):
    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (15,15), 0)
    _, mask = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)
    kernel = np.ones((7,7), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("  [trim_cv] 半紙輪郭が検出できず → 元画像返却")
        return pil_img

    largest_contour = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(largest_contour, 0.02*cv2.arcLength(largest_contour, True), True)

    # 多角形→四角形補正
    if len(approx) == 4:
        pts = approx.reshape(4,2)
    else:
        rect = cv2.minAreaRect(largest_contour)
        pts = cv2.boxPoints(rect)
        pts = np.array(pts, dtype=np.float32)

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    ordered_pts = np.zeros((4,2), dtype=np.float32)
    ordered_pts[0] = pts[np.argmin(s)]
    ordered_pts[2] = pts[np.argmax(s)]
    ordered_pts[1] = pts[np.argmin(diff)]
    ordered_pts[3] = pts[np.argmax(diff)]

    aspect_ratio = 334/244
    output_height = round(output_width * aspect_ratio)
    dst = np.float32([[0,0],[output_width-1,0],[output_width-1,output_height-1],[0,output_height-1]])
    M = cv2.getPerspectiveTransform(ordered_pts, dst)
    warped = cv2.warpPerspective(img_cv, M, (output_width, output_height))

    if debug:
        debug_img = img_cv.copy()
        cv2.drawContours(debug_img, [ordered_pts.astype(int)], -1, (0,0,255), 3)
        debug_pil = Image.fromarray(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
        save_debug_to_dropbox(debug_pil, f"trim_{file_name}_{page_idx}.png")

    return Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))

# ===== A4 パディング =====
def fit_to_a4_padded(pil_img, a4=(2480, 3508)):
    w, h = pil_img.size
    scale = min(a4[0] / w, a4[1] / h)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    resized = pil_img.resize(new_size, Image.LANCZOS)
    canvas = Image.new("RGB", a4, (255,255,255))
    off = ((a4[0]-new_size[0])//2, (a4[1]-new_size[1])//2)
    canvas.paste(resized, off)
    return canvas

# ===== PDF→画像 =====
def pdf_to_images(pdf_bytes):
    images = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for i, page in enumerate(doc):
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    return images

# ===== Dropbox ファイル処理 =====
def process_file(file_metadata):
    file_path = file_metadata.path_lower
    file_name = os.path.basename(file_path)
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    print(f"\n=== 処理開始: {file_name} ===")

    if ext not in SUPPORTED_EXTS:
        fail_dest = f"{DROPBOX_FAILED_FOLDER}/{file_name}"
        try:
            dbx.files_move_v2(file_path, fail_dest, autorename=True)
            print(f"[SKIP] 非対応拡張子 → {fail_dest} に移動")
        except Exception as e:
            print(f"[ERROR] {e}")
        return

    try:
        _, res = dbx.files_download(file_path)
        data = res.content

        if ext == ".pdf":
            images = pdf_to_images(data)
        else:
            img = Image.open(io.BytesIO(data))
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
            images = [img]

        processed_images = []
        for idx, img in enumerate(images):
            # 高精度トリミング
            img_trimmed = trim_paper_cv(img, output_width=2400,
                                        debug=DEBUG_SAVE, file_name=file_name, page_idx=idx)

            # 向き補正
            w, h = img_trimmed.size
            if w > h:
                img_trimmed = img_trimmed.rotate(90, expand=True)

            # A4 パディング
            a4_img = fit_to_a4_padded(img_trimmed)
            processed_images.append(a4_img)

        # PDF にまとめてアップロード
        pdf_bytes = io.BytesIO()
        processed_images[0].save(pdf_bytes, format="PDF", save_all=True, append_images=processed_images[1:])
        pdf_bytes.seek(0)
        dest_pdf_path = f"{DROPBOX_PRINT_FOLDER}/{os.path.splitext(file_name)[0]}.pdf"
        dbx.files_upload(pdf_bytes.read(), dest_pdf_path, mode=dropbox.files.WriteMode("overwrite"))

        # 元ファイルを補正済フォルダへ移動
        processed_dest = f"{DROPBOX_PROCESSED_FOLDER}/{file_name}"
        dbx.files_move_v2(file_path, processed_dest, autorename=True)

        print(f"✅ 成功: {file_name}")

    except Exception as e:
        fail_dest = f"{DROPBOX_FAILED_FOLDER}/{file_name}"
        try:
            dbx.files_move_v2(file_path, fail_dest, autorename=True)
        except:
            pass
        print(f"❌ 失敗: {file_name} ({e})")

# ===== メイン =====
def main():
    res = dbx.files_list_folder(DROPBOX_SRC_FOLDER)
    for entry in res.entries:
        if isinstance(entry, dropbox.files.FileMetadata):
            process_file(entry)

if __name__ == "__main__":
    main()