# disc5_build_docpdf.py
# Renders DISC5_SKANN_Technical_Documentation.md (repo root) into a typeset PDF
# with the four inserted figures. Paths are repo-relative: run from anywhere,
# reads ../DISC5_SKANN_Technical_Documentation.md and ../figures relative to this script.

import re
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph,
                                Spacer, Image, Table, TableStyle, HRFlowable)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

REPO = Path(__file__).resolve().parents[1]
MD_PATH = REPO / "DISC5_SKANN_Technical_Documentation.md"
OUT_PATH = REPO / "DISC5_SKANN_Technical_Documentation.pdf"
FIG_DIR = REPO / "figures"

# ---- fonts (DejaVu for full glyph coverage: arrows, >=, approx, etc.) ----
FD = "/usr/share/fonts/truetype/dejavu/"
pdfmetrics.registerFont(TTFont("DV", FD + "DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DV-B", FD + "DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("DV-I", FD + "DejaVuSans-Oblique.ttf"))
pdfmetrics.registerFont(TTFont("DV-BI", FD + "DejaVuSans-BoldOblique.ttf"))
pdfmetrics.registerFont(TTFont("DV-M", FD + "DejaVuSansMono.ttf"))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")

NAVY = colors.HexColor("#123a5f")
GREY = colors.HexColor("#555555")
LGREY = colors.HexColor("#e9eef4")

S = {
    "title":  ParagraphStyle("title", fontName="DV-B", fontSize=19, leading=24, textColor=NAVY, spaceAfter=6),
    "subtitle": ParagraphStyle("subtitle", fontName="DV", fontSize=11, leading=15, textColor=GREY, spaceAfter=2),
    "h1": ParagraphStyle("h1", fontName="DV-B", fontSize=13.5, leading=17, textColor=NAVY, spaceBefore=14, spaceAfter=5),
    "h2": ParagraphStyle("h2", fontName="DV-B", fontSize=11, leading=14, textColor=NAVY, spaceBefore=10, spaceAfter=3),
    "body": ParagraphStyle("body", fontName="DV", fontSize=9.2, leading=13.2, spaceAfter=5, alignment=4),
    "bullet": ParagraphStyle("bullet", fontName="DV", fontSize=9.2, leading=13.0, spaceAfter=3,
                              leftIndent=14, bulletIndent=4, alignment=4),
    "cell": ParagraphStyle("cell", fontName="DV", fontSize=8.2, leading=10.8),
    "cellh": ParagraphStyle("cellh", fontName="DV-B", fontSize=8.2, leading=10.8, textColor=colors.white),
    "cap": ParagraphStyle("cap", fontName="DV-I", fontSize=8.2, leading=11, textColor=GREY,
                           alignment=1, spaceBefore=2, spaceAfter=8),
}

def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def inline(t):
    t = esc(t)
    t = re.sub(r"`([^`]+)`", r'<font name="DV-M" size="8.3">\1</font>', t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<!\w)\*([^*\n]+)\*(?!\w)", r"<i>\1</i>", t)
    return t

def md_table(rows, width):
    ncols = len(rows[0])
    if ncols == 2:
        cw = [0.34 * width, 0.66 * width]
    else:
        first = 0.30 * width
        cw = [first] + [(width - first) / (ncols - 1)] * (ncols - 1)
    data = [[Paragraph(inline(c), S["cellh"]) for c in rows[0]]]
    for r in rows[1:]:
        data.append([Paragraph(inline(c), S["cell"]) for c in r])
    t = Table(data, colWidths=cw, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LGREY]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b8c4d0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
    ]))
    return t

FIG_WIDTH_CM = {"fig3_pipeline.png": 15.2}  # default 16.4 for the rest

def build(md_path, out_path, width):
    story = []
    story.append(Spacer(1, 0.6 * cm))
    story.append(Paragraph("DISC5 / SKANN \u2014 Technical Documentation", S["title"]))
    story.append(Paragraph("Passive-sonar vessel re-identification: data, preprocessing, augmentation, "
                           "architecture, training, and evaluation", S["subtitle"]))
    story.append(Paragraph("Delivered checkpoint: <b>ft2</b> (disc5_arcface_8k_ft2_ep003.pth) \u00b7 July 2026", S["subtitle"]))
    story.append(Spacer(1, 0.15 * cm))
    story.append(HRFlowable(width="100%", thickness=1.1, color=NAVY, spaceAfter=8))

    lines = open(md_path, encoding="utf-8").read().splitlines()
    i, n = 0, len(lines)
    while i < n:
        ln = lines[i].rstrip()
        mimg = re.match(r"^!\[(.*)\]\((.*)\)$", ln)
        if mimg:
            cap, path = mimg.group(1), mimg.group(2)
            fn = path.split("/")[-1]
            w_cm = FIG_WIDTH_CM.get(fn, 16.4)
            img = Image(str(FIG_DIR / fn))
            scale = (w_cm * cm) / img.imageWidth
            img.drawWidth = w_cm * cm
            img.drawHeight = img.imageHeight * scale
            story.append(Spacer(1, 4)); story.append(img)
            story.append(Paragraph(inline(cap), S["cap"]))
            i += 1; continue
        if not ln:
            i += 1; continue
        if ln == "---":
            story.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#c7d2dd"),
                                    spaceBefore=6, spaceAfter=6))
            i += 1; continue
        if ln.startswith("# "):
            i += 1; continue  # doc title handled by title block
        if ln.startswith("## "):
            story.append(Paragraph(inline(ln[3:]), S["h1"])); i += 1; continue
        if ln.startswith("### "):
            story.append(Paragraph(inline(ln[4:]), S["h2"])); i += 1; continue
        if ln.startswith("> "):
            story.append(Paragraph(inline(ln[2:]), ParagraphStyle(
                "q", parent=S["body"], leftIndent=12, textColor=GREY,
                borderColor=NAVY, borderWidth=0, fontName="DV-I")))
            i += 1; continue
        if ln.startswith("|"):
            rows = []
            while i < n and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            story.append(Spacer(1, 2)); story.append(md_table(rows, width)); story.append(Spacer(1, 6))
            continue
        m = re.match(r"^(\d+)\.\s+(.*)$", ln)
        if m or ln.startswith("- ") or ln.startswith("\u2022 "):
            if m:
                bullet, text = m.group(1) + ".", m.group(2)
            else:
                bullet, text = "\u2022", ln[2:]
            # absorb continuation lines (indented or plain wrap until blank/next structure)
            j = i + 1
            while j < n and lines[j].strip() and not re.match(r"^(\d+\.\s|- |\u2022 |#|\||---$|> |!\[)", lines[j].strip()):
                text += " " + lines[j].strip(); j += 1
            story.append(Paragraph(inline(text), S["bullet"], bulletText=bullet))
            i = j; continue
        # normal paragraph, absorb wraps
        text = ln
        j = i + 1
        while j < n and lines[j].strip() and not re.match(r"^(\d+\.\s|- |\u2022 |#|\||---$|> |!\[)", lines[j].strip()):
            text += " " + lines[j].strip(); j += 1
        story.append(Paragraph(inline(text), S["body"]))
        i = j

    def footer(canv, doc_):
        canv.saveState()
        canv.setFont("DV", 7.5); canv.setFillColor(GREY)
        canv.drawString(2 * cm, 1.05 * cm, "DISC5 / SKANN \u2014 Technical Documentation (ft2)")
        canv.drawRightString(A4[0] - 2 * cm, 1.05 * cm, f"Page {canv.getPageNumber()}")
        canv.setStrokeColor(colors.HexColor("#c7d2dd")); canv.setLineWidth(0.5)
        canv.line(2 * cm, 1.35 * cm, A4[0] - 2 * cm, 1.35 * cm)
        canv.restoreState()

    doc = BaseDocTemplate(str(out_path), pagesize=A4,
                          leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.7 * cm, bottomMargin=1.8 * cm)
    fr = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")
    doc.addPageTemplates([PageTemplate(id="p", frames=[fr], onPage=footer)])
    doc.build(story)

if __name__ == "__main__":
    W = A4[0] - 4 * cm
    build(MD_PATH, OUT_PATH, W)
    print("pdf written:", OUT_PATH)
