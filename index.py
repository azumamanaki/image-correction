# -*- coding: utf-8 -*-
import os
import io
import dropbox
import requests
import numpy as np
from PIL import Image, ImageOps
import cv2
import fitz  # PyMuPDF
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path
import datetime as dt



# ===== 設定（必須環境変数） =====
DROPBOX_CLIENT_ID = os.environ["DROPBOX_CLIENT_ID"]
DROPBOX_CLIENT_SECRET = os.environ["DROPBOX_CLIENT_SECRET"]
DROPBOX_REFRESH_TOKEN = os.environ["DROPBOX_REFRESH_TOKEN"]

BUCKETS = [
    {
        "name": "添削用",
        "input": "/おうち書道/共有データ/【受講生】/【添削用　作品】/提出画像",  # /添削用 作品/提出画像
        "print": "/おうち書道/共有データ/【受講生】/【添削用　作品】/添削用印刷未",  # /添削用印刷未
        "processed": "/おうち書道/共有データ/【受講生】/【添削用　作品】/補正済元画像",  # /補正済元画像
        "failed": "/おうち書道/共有データ/【受講生】/【添削用　作品】/処理失敗",        # /処理失敗
        "target_px": (2480, 3508),
    },
    {
        "name": "清書用-硬筆",
        "input": "/おうち書道/共有データ/【受講生】/【清書用　作品（出品用）】/硬筆/提出画像",
        "print": "/おうち書道/共有データ/【受講生】/【清書用　作品（出品用）】/硬筆/清書用印刷未",
        "processed": "/おうち書道/共有データ/【受講生】/【清書用　作品（出品用）】/硬筆/補正済元画像",
        "failed": "/おうち書道/共有データ/【受講生】/【清書用　作品（出品用）】/硬筆/処理失敗",
        "target_px": (1496, 2083),
    },
    {
        "name": "清書用-毛筆",
        "input": "/おうち書道/共有データ/【受講生】/【清書用　作品（出品用）】/毛筆/提出画像",
        "print": "/おうち書道/共有データ/【受講生】/【清書用　作品（出品用）】/毛筆/清書用印刷未",
        "processed": "/おうち書道/共有データ/【受講生】/【清書用　作品（出品用）】/毛筆/補正済元画像",
        "failed": "/おうち書道/共有データ/【受講生】/【清書用　作品（出品用）】/毛筆/処理失敗",
        "target_px": (2870, 3937),
    },
]

SUPPORTED_EXTS = [".png", ".jpeg", ".jpg", ".pdf"]

# ===== 外周の黒帯除去 =====
DARK_RATIO_EDGE = 0.40
DARK_THRESH     = 90
SAFE_MARGIN_PX  = 2

# ===== 紙マスク =====
HSV_V_MIN  = 150
HSV_S_MAX  = 60
ADAPT_BLOCK = 51
ADAPT_C     = 10
K_CLOSE     = (9, 9)
K_OPEN      = (5, 5)
MIN_AREA_RATIO = 0.20

# ===== 内側の黒枠（外枠）検出：コラム誤検出を抑制 =====
# 反転適応二値化（黒線→白）
ADAPT_BLOCK_INV = 31
ADAPT_C_INV     = 10
FRAME_CONNECT_K = (3, 3)    # 黒線の連結

# 候補矩形の面積・幅・アスペクトのガード
FRAME_MIN_ARATIO   = 0.15   # 面積比の下限（全体の15%以上）
FRAME_MAX_ARATIO   = 0.95   # 上限
FRAME_MIN_W_FRAC   = 0.35   # 画像幅に対する最小幅 35%（細コラム排除）
FRAME_PAPER_AR_MIN = 1.05   # 縦/横（紙っぽい範囲）
FRAME_PAPER_AR_MAX = 1.80

# “細いコラム”の定義（縦長過ぎる候補）
COLUMN_AR_MIN = 3.0         # 縦/横が3以上ならコラム候補
COLUMN_MIN_H_FRAC = 0.6     # 画像高さの60%以上の高さがある細長いもの

# スコアリング
TARGET_ASPECT  = 1.38       # 半紙/硬筆の事前分布
ASPECT_TOL     = 0.60       # 許容（±0.60）
FRAME_CENTER_BIAS = 0.0005  # 中心に近いほど微優先
FRAME_INSIDE_MEAN_MIN = 160 # 内側の明るさ（紙想定）

# 外枠が見つかったら内側に少し寄せる（黒線を避ける）
FRAME_INNER_MARGIN_RATIO = 0.01

# 既存の定数名に合わせてください。無ければ下の2つは任意で定義。
FRAME_INNER_MARGIN_RATIO = globals().get("FRAME_INNER_MARGIN_RATIO", 0.01)
MIN_AREA_RATIO = globals().get("MIN_AREA_RATIO", 0.02)
SAFE_MARGIN_PX = globals().get("SAFE_MARGIN_PX", 2)

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



def ensure_dropbox_folders(dbx, paths: dict):
    for k in ("print","processed","failed"):
        try:
            dbx.files_create_folder_v2(paths[k])
        except Exception:
            # 既に存在すればエラーになるので無視
            pass

# ===== 高精度トリミング =====
# -*- coding: utf-8 -*-
#ほぼ完璧

"""
おうち書道：黒縁/色背景除去 + 内側細枠優先トリミング（コラム誤検出対策版）
+ 仕上げ：四辺の“白さ”で控えめに追いトリム（paper_mask / frame 両ルート対応）

- 入力: fram_image/*.jpg|jpeg|png
- 出力: complete/
"""


def auto_trim_black_edges(img: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    h, w = img.shape[:2]
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    def r(y): return (g[y, :] < DARK_THRESH).mean()
    def c(x): return (g[:, x] < DARK_THRESH).mean()
    top, bot, lef, rig = 0, h-1, 0, w-1
    while top < bot and r(top) > DARK_RATIO_EDGE: top += 1
    while bot > top and r(bot) > DARK_RATIO_EDGE: bot -= 1
    while lef < rig and c(lef) > DARK_RATIO_EDGE: lef += 1
    while rig > lef and c(rig) > DARK_RATIO_EDGE: rig -= 1
    top  = min(max(0, top + SAFE_MARGIN_PX), h-2)
    lef  = min(max(0, lef + SAFE_MARGIN_PX), w-2)
    bot  = max(min(h-1, bot - SAFE_MARGIN_PX), top+1)
    rig  = max(min(w-1, rig - SAFE_MARGIN_PX), lef+1)
    return img[top:bot+1, lef:rig+1], dict(step="edge_trim", top=top, bottom=bot, left=lef, right=rig)

def build_paper_mask(img: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    bright_low_sat = cv2.inRange(hsv, (0,0,HSV_V_MIN), (179,HSV_S_MAX,255))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, ADAPT_BLOCK, ADAPT_C)
    mask = cv2.bitwise_or(bright_low_sat, adapt)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, K_CLOSE), 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, K_OPEN), 1)
    return mask, dict(step="paper_mask")

def largest_cc(mask: np.ndarray) -> Tuple[np.ndarray, float]:
    # ★ 追加：必ず 0/1 の uint8 に
    bw = (mask > 0).astype(np.uint8)

    num, lab, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if num <= 1:
        return np.zeros_like(bw, dtype=np.uint8), 0.0
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + int(np.argmax(areas))
    out = np.zeros_like(bw, dtype=np.uint8)
    out[lab == idx] = 255
    return out, float(areas.max() / bw.size)

def order_quad(pts: np.ndarray) -> np.ndarray:
    pts = pts.astype(np.float32)
    s = pts.sum(axis=1); d = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]; br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]; bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def warp_by_quad(img: np.ndarray, quad: np.ndarray, inner_ratio: float) -> np.ndarray:
    q = order_quad(quad)
    w = int(np.linalg.norm(q[1]-q[0])); h = int(np.linalg.norm(q[3]-q[0]))
    w = max(w,10); h = max(h,10)
    M = cv2.getPerspectiveTransform(q, np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], np.float32))
    warped = cv2.warpPerspective(img, M, (w, h))
    m = int(round(min(w,h)*inner_ratio))
    if m>0 and w>2*m and h>2*m: warped = warped[m:h-m, m:w-m]
    if warped.shape[0] > 2*SAFE_MARGIN_PX and warped.shape[1] > 2*SAFE_MARGIN_PX:
        warped = warped[SAFE_MARGIN_PX:-SAFE_MARGIN_PX, SAFE_MARGIN_PX:-SAFE_MARGIN_PX]
    return warped

def visualize_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    vis = img.copy()
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, (0,0,255), 2)
    return vis


# ===== 仕上げ：四辺の“白さ”で控えめに追いトリム =====
def measure_edge_white_ratio(img: np.ndarray, band_frac: float = 0.02, thr: int = 225) -> Dict[str, float]:
    """画像の四辺バンドの『白さ(=明るさ)比率』を返す"""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    b = max(2, int(min(h, w) * band_frac))

    def ratio(region):
        return float((region >= thr).mean())  # 0..1

    r = {
        "top":    ratio(g[0:b, :]),
        "bottom": ratio(g[h-b:h, :]),
        "left":   ratio(g[:, 0:b]),
        "right":  ratio(g[:, w-b:w]),
    }
    return r

def suggest_insets_from_ratios(r: Dict[str, float], w: int, h: int) -> Dict[str, int]:
    """
    4辺が低い場合でも“特に低い辺”は控えめに寄せる。
    通常は段階テーブルに従い寄せる。
    """
    vals = [r.get("top",1), r.get("bottom",1), r.get("left",1), r.get("right",1)]
    names = ["top","bottom","left","right"]

    base = max(4, int(min(w, h) * 0.015))  # 控えめ基準
    tiers = [(0.98,0.0),(0.95,0.5),(0.90,1.0),(0.80,1.6),(0.70,2.2),(0.60,2.8),(0.00,3.4)]
    def inset_for(val: float) -> int:
        for th,k in tiers:
            if val >= th: return int(round(base*k))
        return int(round(base*tiers[-1][1]))

    ins = {k:0 for k in names}
    all_low = all(v <= 0.50 for v in vals)
    if all_low:
        anchor = float(np.median(vals))
        def is_much_lower(v): return (v <= min(0.45, anchor - 0.07))
        for k,v in zip(names, vals):
            ins[k] = int(round(inset_for(v) * (0.8 if is_much_lower(v) else 0.0)))
    else:
        ins["top"]    = inset_for(r["top"])
        ins["bottom"] = inset_for(r["bottom"])
        ins["left"]   = inset_for(r["left"])
        ins["right"]  = inset_for(r["right"])

    # 上限キャップ（過切り防止）
    v_cap = max(6, min(18, int(h * 0.06)))  # 上下
    h_cap = max(6, min(14, int(w * 0.05)))  # 左右
    ins["top"]    = min(ins["top"], v_cap)
    ins["bottom"] = min(ins["bottom"], v_cap)
    ins["left"]   = min(ins["left"], h_cap)
    ins["right"]  = min(ins["right"], h_cap)
    return {k:int(v) for k,v in ins.items()}


# ===== 内側枠検出（コラム耐性） =====
def detect_frame_or_compose(img: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    - 黒線強調（反転適応二値化）→連結→輪郭抽出
    - '紙っぽい'矩形のみを強く優先（AR/幅/面積）
    - 縦長コラムは除外。ただしコラムが複数ある場合はまとめて外接矩形へ合成して再判定
    """
    h, w = img.shape[:2]
    img_area = float(h*w)
    cx, cy = w/2.0, h/2.0

    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (5,5), 0)
    inv = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, ADAPT_BLOCK_INV, ADAPT_C_INV)
    inv = cv2.morphologyEx(inv, cv2.MORPH_CLOSE,
                           cv2.getStructuringElement(cv2.MORPH_RECT, FRAME_CONNECT_K), 2)

    cnts, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    paperish: List[Dict[str,Any]] = []
    columns:  List[Dict[str,Any]] = []

    for c in cnts:
        peri = cv2.arcLength(c, True)
        if peri < 0.3*(h+w): continue
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        if len(approx)!=4 or not cv2.isContourConvex(approx): continue

        quad = order_quad(approx.reshape(-1,2))
        ww = np.linalg.norm(quad[1]-quad[0]); hh = np.linalg.norm(quad[3]-quad[0])
        if ww<=0 or hh<=0: continue
        aratio = (ww*hh)/img_area
        ar = hh/ww
        w_frac = ww / w
        if aratio<0.02:  # ごく小さいノイズ除去
            continue

        # 内側の明るさチェック
        mask = np.zeros((h,w), np.uint8)
        cv2.fillConvexPoly(mask, quad.astype(np.int32), 255)
        inner = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (15,15)))
        inside_mean = cv2.mean(g, mask=inner)[0]

        rec = dict(quad=quad, ar=ar, w_frac=float(w_frac),
                   aratio=float(aratio), peri=float(peri), inside_mean=float(inside_mean))

        # 細長コラム？
        if (ar >= COLUMN_AR_MIN and hh/h >= COLUMN_MIN_H_FRAC):
            columns.append(rec)
            continue

        # “紙っぽい”条件
        if (FRAME_MIN_ARATIO <= aratio <= FRAME_MAX_ARATIO and
            FRAME_PAPER_AR_MIN <= ar <= FRAME_PAPER_AR_MAX and
            w_frac >= FRAME_MIN_W_FRAC and
            inside_mean >= FRAME_INSIDE_MEAN_MIN):
            # スコア：面積大・中心近い・目標アスペクトに近い
            qcx,qcy = quad.mean(axis=0)
            dist = ((qcx-cx)**2 + (qcy-cy)**2)**0.5
            aspect_penalty = abs(ar - TARGET_ASPECT) / ASPECT_TOL  # 小さいほど良い
            score = aratio - FRAME_CENTER_BIAS*dist - 0.15*aspect_penalty
            rec["score"] = float(score)
            paperish.append(rec)

    info = dict(step="inner_frame_detect_v3",
                paperish_candidates=len(paperish),
                column_candidates=len(columns))

    best = None
    if paperish:
        best = max(paperish, key=lambda r: r["score"])
    else:
        # コラムが複数並んでいる場合は外接矩形に“合成”
        if len(columns) >= 2:
            xs = []; ys = []
            for r in columns:
                q = r["quad"]
                xs.extend([q[:,0].min(), q[:,0].max()])
                ys.extend([q[:,1].min(), q[:,1].max()])
            x0, x1 = max(0, int(min(xs))), min(w-1, int(max(xs)))
            y0, y1 = max(0, int(min(ys))), min(h-1, int(max(ys)))
            # 合成矩形を紙っぽいか再チェック
            ww = x1 - x0 + 1; hh = y1 - y0 + 1
            if ww>10 and hh>10:
                ar = hh/ww; aratio = (ww*hh)/img_area; w_frac = ww/w
                if (FRAME_MIN_ARATIO <= aratio <= FRAME_MAX_ARATIO and
                    FRAME_PAPER_AR_MIN <= ar <= FRAME_PAPER_AR_MAX and
                    w_frac >= FRAME_MIN_W_FRAC):
                    quad = np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]], np.float32)
                    qcx,qcy = quad.mean(axis=0)
                    dist = ((qcx-cx)**2 + (qcy-cy)**2)**0.5
                    aspect_penalty = abs(ar - TARGET_ASPECT) / ASPECT_TOL
                    score = aratio - FRAME_CENTER_BIAS*dist - 0.15*aspect_penalty
                    best = dict(quad=quad, ar=ar, w_frac=w_frac,
                                aratio=aratio, inside_mean=999, score=float(score))
    if best is not None:
        info.update(dict(found=True,
                         quad=best["quad"].tolist(),
                         ar=float(best["ar"]), w_frac=float(best["w_frac"]),
                         aratio=float(best["aratio"]), score=float(best["score"])))
        return best["quad"].astype(np.float32), info
    else:
        info.update(dict(found=False))
        return None, info

def trim_shodo_paper(pil_img: Image.Image) -> Image.Image:
    bgr0 = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    bgr1, _info_edge = auto_trim_black_edges(bgr0)

    # detect は (quad, info) で返るので素直にアンパック
    quad, _info_frame = detect_frame_or_compose(bgr1)

    if quad is not None:
        out = warp_by_quad(bgr1, quad.astype(np.float32), inner_ratio=FRAME_INNER_MARGIN_RATIO)

        # ★ frame ルートでも仕上げ寄せ（process_one と同じ）
        ratios = measure_edge_white_ratio(out, band_frac=0.02, thr=225)
        ih, iw = out.shape[:2]
        inset  = suggest_insets_from_ratios(ratios, iw, ih)
        ty, by = inset["top"], inset["bottom"]
        lx, rx = inset["left"], inset["right"]
        if ty+by < ih-4 and lx+rx < iw-4:
            out = out[ty:ih-by, lx:iw-rx]

    else:
        mret = build_paper_mask(bgr1)
        mask = mret[0] if (isinstance(mret, tuple) and len(mret) >= 1) else mret

        lcc, area_ratio = largest_cc(mask)
        if area_ratio >= MIN_AREA_RATIO and lcc.max() > 0:
            ys, xs = np.where(lcc > 0)
            y0, y1 = int(ys.min()), int(ys.max())
            x0, x1 = int(xs.min()), int(xs.max())
            H, W = bgr1.shape[:2]
            x0 = max(0, x0 + SAFE_MARGIN_PX); y0 = max(0, y0 + SAFE_MARGIN_PX)
            x1 = min(W-1, x1 - SAFE_MARGIN_PX); y1 = min(H-1, y1 - SAFE_MARGIN_PX)
            out = bgr1[y0:y1+1, x0:x1+1]

            # ★ こちらも仕上げ寄せ
            ratios = measure_edge_white_ratio(out, band_frac=0.02, thr=225)
            ih, iw = out.shape[:2]
            inset  = suggest_insets_from_ratios(ratios, iw, ih)
            ty, by = inset["top"], inset["bottom"]
            lx, rx = inset["left"], inset["right"]
            if ty+by < ih-4 and lx+rx < iw-4:
                out = out[ty:ih-by, lx:iw-rx]
        else:
            out = bgr1

    return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))


# ===== リサイズ（余白なし） =====
def resize_to_target(pil_img: Image.Image, target_px: tuple[int,int]) -> Image.Image:
    """
    画像を強制的に target_px サイズにリサイズする（余白なし）
    """
    return pil_img.resize(target_px, Image.LANCZOS)

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
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".pdf"}

def process_file(file_metadata, paths: dict):
    """
    入力: PDF/JPG/PNG
    処理: 画像化 → トリム → （必要なら回転）→ A4等のターゲット解像度に整形
    出力: PNGで paths['print'] へ保存（複数ページは _p1.png, _p2.png ...）
    その後、元ファイルを paths['processed']/<YYYY-M>/ へ移動
    """
    file_path = file_metadata.path_lower
    file_name = os.path.basename(file_path)
    stem, ext = os.path.splitext(file_name)
    ext = ext.lower()

    print(f"\n=== 処理開始: {file_name} ===")

    if ext not in SUPPORTED_EXTS:
        fail_dest = f"{paths['failed']}/{file_name}"
        try:
            dbx.files_move_v2(file_path, fail_dest, autorename=True)
            print(f"[SKIP] 非対応拡張子 → {fail_dest}")
        except Exception as e:
            print(f"[ERROR] move(non-supported): {e}")
        return

    try:
        # 1) ダウンロード
        _, res = dbx.files_download(file_path)
        data = res.content

        # 2) 画像化
        if ext == ".pdf":
            images = pdf_to_images(data)  # bytes -> [PIL.Image,...]
            if not images:
                raise ValueError("PDFページが0件")
        else:
            img = Image.open(io.BytesIO(data))
            img = ImageOps.exif_transpose(img).convert("RGB")
            images = [img]

        # 3) ページ処理
        processed_images = []
        for idx, pil_img in enumerate(images, start=1):
            trimmed = trim_shodo_paper(pil_img)

            # 縦長に統一
            w, h = trimmed.size
            if w > h:
                trimmed = trimmed.rotate(90, expand=True)

            a4_img = resize_to_target(trimmed, paths["target_px"])
            processed_images.append((idx, a4_img))

        # 4) PNGでアップロード
        uploaded = []
        for idx, out_img in processed_images:
            out_name = stem if len(processed_images) == 1 else f"{stem}_p{idx}"
            out_png_path = f"{paths['print']}/{out_name}.png"

            buf = io.BytesIO()
            out_img.save(buf, format="PNG", optimize=True, compress_level=1)
            buf.seek(0)

            dbx.files_upload(buf.read(), out_png_path,
                             mode=dropbox.files.WriteMode("overwrite"))
            uploaded.append(out_png_path)

        print(f"✅ PNG保存: {len(uploaded)}ファイル -> print")

        # 5) 元ファイルを processed/<YYYY-M>/ へ移動（フォルダが無ければ作成）
        now = dt.datetime.now()
        ym_bucket = f"{now.year}-{now.month}"   # 例: 2025-9
        processed_month_dir = f"{paths['processed']}/{ym_bucket}"

        # フォルダ作成（存在してもOK）
        try:
            dbx.files_create_folder_v2(processed_month_dir)
        except Exception:
            pass

        processed_dest = f"{processed_month_dir}/{file_name}"
        dbx.files_move_v2(file_path, processed_dest, autorename=True)
        print(f"→ processed へ移動: {processed_dest}")

    except Exception as e:
        # 失敗時に failed へ
        fail_dest = f"{paths['failed']}/{file_name}"
        try:
            dbx.files_move_v2(file_path, fail_dest, autorename=True)
        except Exception as me:
            print(f"[ERROR] move(failed) も失敗: {me}")
        print(f"❌ 失敗: {file_name} ({e})")

# ===== メイン =====
def main():
    for bucket in BUCKETS:
        print(f"\n=== バケット開始: {bucket['name']} ===")
        ensure_dropbox_folders(dbx, bucket)  # 任意

        # 提出画像フォルダ直下のみ（サブフォルダも対象にするなら recursive=True）
        res = dbx.files_list_folder(bucket["input"])
        entries = list(res.entries)
        while res.has_more:
            res = dbx.files_list_folder_continue(res.cursor)
            entries.extend(res.entries)

        count = 0
        for entry in entries:
            if isinstance(entry, dropbox.files.FileMetadata):
                process_file(entry, bucket)
                count += 1
        print(f"[INFO] {bucket['name']}: {count} 件処理")

if __name__ == "__main__":
    main()