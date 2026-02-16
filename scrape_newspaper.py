#!/usr/bin/env python3
"""
Extract the first page of each WSJ PDF as a WebP image.

Outputs:
  - front_pages/<pdf_name>.webp
"""

import io
import os
import sys
import fitz  # PyMuPDF
from PIL import Image

PDF_DIR = os.path.join(os.path.dirname(__file__), "Daily Newspapers")
OUT_DIR = os.path.join(os.path.dirname(__file__), "front_pages")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    pdfs = sorted(f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf"))
    if not pdfs:
        print("No PDFs found in", PDF_DIR)
        sys.exit(1)

    for pdf_file in pdfs:
        pdf_path = os.path.join(PDF_DIR, pdf_file)
        pdf_name = os.path.splitext(pdf_file)[0]
        out_path = os.path.join(OUT_DIR, f"{pdf_name}.webp")

        try:
            doc = fitz.open(pdf_path)
            page = doc[0]
            # Render at 2x zoom for good quality, save as webp via Pillow
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            img.save(out_path, "WEBP", quality=85)
            doc.close()
            print(f"OK  {pdf_name}.webp")
        except Exception as e:
            print(f"ERR {pdf_file}: {e}")

    print(f"\nDone. {len(pdfs)} pages saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
