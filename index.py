# -*- coding: utf-8 -*-
import os
import io
import dropbox
import requests
import numpy as np
from PIL import Image, ImageOps
import cv2
import fitz  # PyMuPDF
from typing import Tuple, Optional
from pathlib import Path


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

# 調整用定数（必要に応じて微調整可）
DARK_THRESH = 40
DARK_RATIO_EDGE = 0.60
SAFE_MARGIN_PX = 2
TARGET_HW = 334/244
ASPECT_TOL = 0.35
FRAME_MIN_AR = 1000
FRAME_INNER_MARGIN_RATIO = 0.01
HSV_V_MIN = 180
HSV_S_MAX = 80
ADAPT_BLOCK = 51
ADAPT_C = 5
MIN_AREA_RATIO = 0.02
EDGE_BAND_FRAC = 0.02
EDGE_WHITE_THR = 225

def _pil_to_bgr(pil: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)

def _bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

def _auto_trim_black_edges(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    top, bottom, left, right = 0, h-1, 0, w-1
    def rd(line): return (line < DARK_THRESH).mean()
    while top < bottom and rd(gray[top])   >= DARK_RATIO_EDGE: top   += 1
    while bottom > top   and rd(gray[bottom])>= DARK_RATIO_EDGE: bottom-= 1
    while left < right  and rd(gray[:,left])>= DARK_RATIO_EDGE: left  += 1
    while right > left  and rd(gray[:,right])>= DARK_RATIO_EDGE: right -= 1
    top = max(0, top - SAFE_MARGIN_PX); left = max(0, left - SAFE_MARGIN_PX)
    bottom = min(h-1, bottom + SAFE_MARGIN_PX); right = min(w-1, right + SAFE_MARGIN_PX)
    return bgr[top:bottom+1, left:right+1]

def _order_quad(pts: np.ndarray) -> np.ndarray:
    pts = pts.astype(np.float32)
    s = pts.sum(axis=1); d = np.diff(pts, axis=1)[:,0]
    ordered = np.zeros((4,2), np.float32)
    ordered[0] = pts[np.argmin(s)]    # TL
    ordered[2] = pts[np.argmax(s)]    # BR
    ordered[1] = pts[np.argmin(d)]    # TR
    ordered[3] = pts[np.argmax(d)]    # BL
    return ordered

def _warp_by_quad(bgr: np.ndarray, quad: np.ndarray, inner_ratio: float=0.0) -> np.ndarray:
    q = _order_quad(quad)
    if inner_ratio > 0:
        cx, cy = q.mean(axis=0)
        q = (q - [cx, cy]) * (1.0 - inner_ratio*2.0) + [cx, cy]
    tl,tr,br,bl = q
    W = int(round(max(np.linalg.norm(br-bl), np.linalg.norm(tr-tl))))
    H = int(round(max(np.linalg.norm(tr-br), np.linalg.norm(tl-bl))))
    W = max(W,16); H = max(H,16)
    dst = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], np.float32)
    M = cv2.getPerspectiveTransform(q, dst)
    return cv2.warpPerspective(bgr, M, (W,H), flags=cv2.INTER_LINEAR)

def _score_rect(cnt: np.ndarray, shape: Tuple[int,int]) -> Optional[Tuple[float,np.ndarray]]:
    if cv2.contourArea(cnt) < FRAME_MIN_AR: return None
    rect = cv2.minAreaRect(cnt); box = cv2.boxPoints(rect).astype(np.float32)
    (cx,cy),(w,h),_ = rect
    if w<=1 or h<=1: return None
    r = max(w,h)/min(w,h)
    pen = abs(r - TARGET_HW)
    if pen > ASPECT_TOL: return None
    H,W = shape[:2]; dc = np.hypot(cx-W/2, cy-H/2)/max(W,H)
    area = w*h
    score = (area*1e-4) - (pen*2.0) - (dc*0.5)
    return (score, box)

def _detect_inner_frame(bgr: np.ndarray) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(gray, 60, 180)
    edges = cv2.dilate(edges, np.ones((3,3),np.uint8), iterations=1)
    contours,_ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in contours:
        approx = cv2.approxPolyDP(c, 0.02*cv2.arcLength(c, True), True)
        if len(approx) < 4: continue
        s = _score_rect(c, bgr.shape)
        if s is None: continue
        if (best is None) or (s[0] > best[0]): best = s
    return None if best is None else best[1]

def _build_paper_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV); h,s,v = cv2.split(hsv)
    m1 = ((v>=HSV_V_MIN) & (s<=HSV_S_MAX)).astype(np.uint8)*255
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    m2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, ADAPT_BLOCK, ADAPT_C)
    mask = cv2.bitwise_or(m1, m2)
    ff = mask.copy(); cv2.floodFill(ff, None, (0,0), 0)
    paper = cv2.bitwise_and(mask, cv2.bitwise_not(ff))
    k = np.ones((5,5), np.uint8)
    paper = cv2.morphologyEx(paper, cv2.MORPH_CLOSE, k, iterations=1)
    paper = cv2.morphologyEx(paper, cv2.MORPH_OPEN,  k, iterations=1)
    return paper

def _largest_cc(mask: np.ndarray) -> Tuple[np.ndarray, float]:
    num, labels = cv2.connectedComponents(mask>0)
    if num <= 1: return (np.zeros_like(mask), 0.0)
    areas = [(labels==i).sum() for i in range(1,num)]
    idx = int(np.argmax(areas))+1
    lcc = (labels==idx).astype(np.uint8)*255
    ratio = areas[idx-1]/mask.size
    return (lcc, float(ratio))

def _measure_edge_white_ratios(bgr: np.ndarray, band_frac: float, thr: int) -> dict:
    h,w = bgr.shape[:2]
    bw = max(1, int(round(w*band_frac))); bh = max(1, int(round(h*band_frac)))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    top=gray[:bh,:]; bottom=gray[h-bh:,:]; left=gray[:,:bw]; right=gray[:,w-bw:]
    f = lambda a: float((a>=thr).mean())
    return {"top":f(top),"bottom":f(bottom),"left":f(left),"right":f(right)}

def _suggest_insets_from_ratios(r: dict, w: int, h: int) -> dict:
    def px(frac,size): return int(round(size*frac))
    def g(v,size): 
        if v<0.25: return px(0.02,size)
        if v<0.50: return px(0.01,size)
        return 0
    return {"top":g(r["top"],h),"bottom":g(r["bottom"],h),"left":g(r["left"],w),"right":g(r["right"],w)}

def trim_shodo_paper(pil_img: Image.Image) -> Image.Image:
    """保存・デバッグ一切なし。PIL→PILで返す。"""
    bgr0 = _pil_to_bgr(pil_img)
    bgr1 = _auto_trim_black_edges(bgr0)
    quad = _detect_inner_frame(bgr1)
    if quad is not None:
        out = _warp_by_quad(bgr1, quad, inner_ratio=FRAME_INNER_MARGIN_RATIO)
    else:
        mask = _build_paper_mask(bgr1)
        lcc, area_ratio = _largest_cc(mask)
        if area_ratio >= MIN_AREA_RATIO and lcc.max()>0:
            ys,xs = np.where(lcc>0)
            y0,y1 = int(ys.min()), int(ys.max())
            x0,x1 = int(xs.min()), int(xs.max())
            H,W = bgr1.shape[:2]
            x0 = max(0, x0+SAFE_MARGIN_PX); y0 = max(0, y0+SAFE_MARGIN_PX)
            x1 = min(W-1, x1-SAFE_MARGIN_PX); y1 = min(H-1, y1-SAFE_MARGIN_PX)
            out = bgr1[y0:y1+1, x0:x1+1]
        else:
            out = bgr1
    # 仕上げの控えめインセット
    h,w = out.shape[:2]
    ratios = _measure_edge_white_ratios(out, EDGE_BAND_FRAC, EDGE_WHITE_THR)
    inset  = _suggest_insets_from_ratios(ratios, w, h)
    ty,by,lx,rx = inset["top"], inset["bottom"], inset["left"], inset["right"]
    if ty+by < h-4 and lx+rx < w-4:
        out = out[ty:h-by, lx:w-rx]
    return _bgr_to_pil(out)
# ==== ここまで貼り付け ====
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
            img_trimmed = trim_shodo_paper(img)

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