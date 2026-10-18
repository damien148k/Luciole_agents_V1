"""OCR : contrat de moteur, rendu de page, implémentation EasyOCR.

Le parser PDF ne connaît que le protocole ``OcrEngine`` : il lui donne
l'image d'une page et reçoit des lignes avec leur boîte et leur confiance.
Les tests utilisent un moteur factice ; EasyOCR (qui tire PyTorch) n'est
importé que si on l'instancie. Contrairement à V5, boîtes et confiances
sont conservées jusque dans les nœuds du document.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pymupdf
from loguru import logger

from luciole.core.document import BBox


@dataclass(frozen=True)
class OcrLine:
    """Une ligne reconnue. ``bbox`` est en pixels de l'image fournie."""

    text: str
    bbox: BBox
    confidence: float


@dataclass(frozen=True)
class RenderedPage:
    png: bytes
    scale: float  # pixels par point PDF (dpi / 72)
    width: int
    height: int


class OcrEngine(Protocol):
    name: str

    def recognize(self, image_png: bytes) -> list[OcrLine]: ...


def render_page(page: pymupdf.Page, dpi: int = 300) -> RenderedPage:
    """Rend une page en PNG à la résolution demandée."""
    scale = dpi / 72
    pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale))
    return RenderedPage(png=pix.tobytes("png"), scale=scale, width=pix.width, height=pix.height)


def to_points(bbox: BBox, scale: float) -> BBox:
    """Convertit une boîte en pixels vers les points PDF de la page."""
    return BBox(x0=bbox.x0 / scale, y0=bbox.y0 / scale, x1=bbox.x1 / scale, y1=bbox.y1 / scale)


def group_lines(lines: list[OcrLine], gap_factor: float = 1.6) -> list[list[OcrLine]]:
    """Regroupe des lignes OCR en paragraphes : une nouvelle ligne rejoint le
    groupe courant si l'écart vertical avec la précédente reste inférieur à
    ``gap_factor`` fois la hauteur de ligne. Lignes triées par position."""
    ordered = sorted(lines, key=lambda ln: (round(ln.bbox.y0), ln.bbox.x0))
    groups: list[list[OcrLine]] = []
    for ln in ordered:
        if groups:
            prev = groups[-1][-1]
            height = max(prev.bbox.y1 - prev.bbox.y0, 1.0)
            if ln.bbox.y0 - prev.bbox.y1 < gap_factor * height:
                groups[-1].append(ln)
                continue
        groups.append([ln])
    return groups


def _easyocr_model_dir() -> str | None:
    """Dossier des modèles EasyOCR pour le mode hors ligne, s'il existe."""
    candidates = []
    if os.environ.get("EASYOCR_MODEL_DIR"):
        candidates.append(Path(os.environ["EASYOCR_MODEL_DIR"]))
    candidates += [Path("/app/models/easyocr"), Path.home() / ".EasyOCR" / "model"]
    for d in candidates:
        if (d / "craft_mlt_25k.pth").exists():
            return str(d)
    return None


class EasyOcrEngine:
    """EasyOCR, chargé depuis le cache local quand il existe (hors ligne)."""

    name = "easyocr"

    def __init__(
        self,
        languages: tuple[str, ...] = ("fr", "en"),
        gpu: bool = True,
        min_confidence: float = 0.3,
    ) -> None:
        try:
            import easyocr  # type: ignore[import-not-found]
        except ImportError as e:  # pragma: no cover - dépend de l'installation
            raise RuntimeError(
                "EasyOCR n'est pas installé : installer l'extra `ocr` "
                "(uv sync --extra ocr) sur une machine avec le profil matériel voulu"
            ) from e
        import numpy as np  # type: ignore[import-not-found]
        from PIL import Image  # type: ignore[import-not-found]

        self._np = np
        self._Image = Image
        self.min_confidence = min_confidence
        model_dir = _easyocr_model_dir()
        kwargs = {"gpu": gpu, "verbose": False}
        if model_dir:
            kwargs |= {"model_storage_directory": model_dir, "download_enabled": False}
            logger.info(f"EasyOCR hors ligne depuis {model_dir}")
        else:
            logger.warning("Modèles EasyOCR introuvables localement : téléchargement nécessaire")
        self._reader = easyocr.Reader(list(languages), **kwargs)

    def recognize(self, image_png: bytes) -> list[OcrLine]:
        import io

        image = self._np.array(self._Image.open(io.BytesIO(image_png)))
        out: list[OcrLine] = []
        for quad, text, conf in self._reader.readtext(image):
            if conf < self.min_confidence or not text.strip():
                continue
            xs = [p[0] for p in quad]
            ys = [p[1] for p in quad]
            out.append(
                OcrLine(
                    text=text.strip(),
                    bbox=BBox(x0=min(xs), y0=min(ys), x1=max(xs), y1=max(ys)),
                    confidence=float(conf),
                )
            )
        return out
