"""Local gallery demo. SQLite stores images and normalized embeddings atomically."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import sqlite3
import threading

import numpy as np
from PIL import Image, ImageOps

PREPROCESS = "rgb-exif-bicubic224-normalize0.5-v1"


def image_data(value):
    """Return RGB image, decoded-pixel identity, and portable PNG bytes."""
    if value is None:
        raise ValueError("Upload a tiger image first.")
    if isinstance(value, Image.Image):
        im = ImageOps.exif_transpose(value).convert("RGB")
    else:
        with Image.open(value) as source:
            im = ImageOps.exif_transpose(source).convert("RGB")
    digest = hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()
    stream = io.BytesIO()
    im.save(stream, format="PNG")
    return im, digest, stream.getvalue()


def normalized(vector):
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    length = np.linalg.norm(vector)
    if len(vector) != 128 or not np.isfinite(vector).all() or length <= 0:
        raise ValueError("Expected a finite, nonzero 128-dimensional embedding.")
    return np.asarray(vector / length, dtype="<f4")


class Encoder:
    def __init__(self, checkpoint, device="auto"):
        import torch
        from torchvision import transforms as T
        from .model import TigerViT
        from .data import sha256

        self.checkpoint = Path(checkpoint).resolve()
        self.fingerprint = sha256(self.checkpoint)
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else torch.device(device)
        )
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError(
                "CUDA unavailable. Use --device cpu or your CUDA PyTorch environment."
            )
        state = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        if (
            not isinstance(state, dict)
            or not {"model", "config", "manifest_sha256"} <= state.keys()
        ):
            raise ValueError(
                "Use the study's best.pt checkpoint, not a pickled model or last.pt."
            )
        self.manifest_hash = state["manifest_sha256"]
        self.config = state["config"]
        self.model = TigerViT(
            pretrained=False, blocks=self.config["settings"]["blocks"]
        )
        self.model.load_state_dict(state["model"], strict=True)
        self.model.to(self.device).eval()
        self.transform = T.Compose(
            [
                T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )

    def __call__(self, image):
        import torch

        with torch.inference_mode():
            vector = self.model(self.transform(image).unsqueeze(0).to(self.device))
        return normalized(vector[0].cpu().numpy())


class Gallery:
    def __init__(self, path, encoder):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.encoder = encoder
        self.lock = threading.RLock()
        expected = json.dumps(
            {
                "schema": 1,
                "checkpoint": encoder.fingerprint,
                "preprocess": PREPROCESS,
                "dimension": 128,
            },
            sort_keys=True,
        )
        with self.connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            old = db.execute(
                "SELECT value FROM metadata WHERE key='signature'"
            ).fetchone()
            if old and old[0] != expected:
                raise ValueError(
                    "Gallery belongs to another checkpoint/preprocessing version. Use a new --database path and re-import images."
                )
            db.execute(
                "INSERT OR IGNORE INTO metadata VALUES ('signature', ?)", (expected,)
            )
            db.execute("""CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY, identity TEXT NOT NULL, digest TEXT UNIQUE NOT NULL,
                image BLOB NOT NULL, embedding BLOB NOT NULL, source TEXT NOT NULL)""")

    def connect(self):
        # Context-managed wrapper closes connections on Windows as well as committing.
        from contextlib import contextmanager

        @contextmanager
        def connection():
            db = sqlite3.connect(self.path, timeout=60)
            try:
                with db:
                    yield db
            finally:
                db.close()

        return connection()

    def inventory(self):
        with self.connect() as db:
            return [
                list(row)
                for row in db.execute(
                    "SELECT identity, COUNT(*) FROM images GROUP BY identity ORDER BY identity"
                )
            ]

    @staticmethod
    def valid_id(identity):
        identity = str(identity).strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", identity):
            raise ValueError(
                "ID must be 1–64 letters, numbers, underscores or hyphens, starting with a letter/number."
            )
        return identity

    def insert(self, db, identity, value, source):
        identity = self.valid_id(identity)
        im, digest, png = image_data(value)
        old = db.execute(
            "SELECT identity FROM images WHERE digest=?", (digest,)
        ).fetchone()
        if old:
            if old[0] != identity:
                raise ValueError(
                    f"This image already belongs to {old[0]}; resolve the label before adding it."
                )
            return False
        vector = normalized(self.encoder(im))
        db.execute(
            "INSERT INTO images(identity,digest,image,embedding,source) VALUES(?,?,?,?,?)",
            (identity, digest, png, vector.tobytes(), source),
        )
        return True

    def enroll(self, value, identity, mode, confirmed):
        if not confirmed:
            raise ValueError(
                "Confirm that you reviewed the matches and identity before saving."
            )
        identity = self.valid_id(identity)
        with self.lock, self.connect() as db:
            exists = db.execute(
                "SELECT 1 FROM images WHERE identity=?", (identity,)
            ).fetchone()
            if mode == "New tiger" and exists:
                raise ValueError(
                    "That ID already exists. Choose 'Existing tiger' to add a reference."
                )
            if mode == "Existing tiger" and not exists:
                raise ValueError(
                    "That ID does not exist. Refresh the gallery or choose 'New tiger'."
                )
            if mode not in {"New tiger", "Existing tiger"}:
                raise ValueError("Invalid enrollment mode.")
            added = self.insert(db, identity, value, "manual enrollment")
        return (
            f"Saved reference for {identity}."
            if added
            else "This reference is already registered; nothing changed."
        )

    def import_manifest(
        self, manifest_path, root, selection, excluded="", progress=None
    ):
        from .data import load_manifest, sha256

        manifest, root, digest = load_manifest(manifest_path, root)
        if digest != self.encoder.manifest_hash:
            raise ValueError(
                "Manifest differs from the one used by this checkpoint. Restore the original manifest unchanged."
            )
        if selection not in {"train", "test-gallery"}:
            raise ValueError("Select train or test-gallery.")
        omit = {x.strip() for x in excluded.split(",") if x.strip()}
        candidates = [
            r
            for r in manifest["records"]
            if (
                r["split"] == "train"
                if selection == "train"
                else r["split"] == "test" and r["role"] == "gallery"
            )
        ]
        missing = omit - {str(r["identity"]) for r in candidates}
        if missing:
            raise ValueError(
                "Excluded IDs are not in this selection: " + ", ".join(sorted(missing))
            )
        rows = [r for r in candidates if str(r["identity"]) not in omit]
        if not rows:
            raise ValueError("No reference images selected.")
        count = 0
        with self.lock, self.connect() as db:
            # Exclusion must never silently claim an already registered ID is unknown.
            registered = {
                r[0] for r in db.execute("SELECT DISTINCT identity FROM images")
            }
            if omit & registered:
                raise ValueError(
                    "An excluded ID is already registered. Delete it first or use a fresh database."
                )
            for i, row in enumerate(rows):
                if progress:
                    progress(
                        i / len(rows), desc=f"Embedding reference {i + 1}/{len(rows)}"
                    )
                path = root / row["path"]
                if sha256(path) != row["sha256"]:
                    raise ValueError("Reference image changed: " + row["path"])
                count += self.insert(
                    db, str(row["identity"]), path, f"{selection}: {row['path']}"
                )
            if progress:
                progress(1, desc="Gallery ready")
        return f"Imported {count} new references; {len(rows) - count} duplicates skipped. Excluded IDs: {', '.join(sorted(omit)) or 'none'}."

    def identify(self, value, threshold):
        threshold = float(threshold)
        if not np.isfinite(threshold) or not -1 <= threshold <= 1:
            raise ValueError("Threshold must be between -1 and 1.")
        im, digest, _ = image_data(value)
        with self.lock:
            with self.connect() as db:
                rows = db.execute(
                    "SELECT identity,digest,image,embedding FROM images ORDER BY id"
                ).fetchall()
            if not rows:
                raise ValueError(
                    "Gallery is empty. Import references or register a tiger first."
                )
            query = normalized(self.encoder(im))
        scores = np.clip(
            np.stack([np.frombuffer(r[3], dtype="<f4") for r in rows]) @ query, -1, 1
        )
        best = {}
        for row, score in zip(rows, scores):
            if row[0] not in best or score > best[row[0]][0]:
                best[row[0]] = (float(score), row[2])
        top = sorted(best.items(), key=lambda pair: (-pair[1][0], pair[0]))[:3]
        accepted = top[0][1][0] >= threshold
        message = (
            f"Candidate match: {top[0][0]}"
            if accepted
            else "No confident match — possibly an unregistered tiger."
        )
        message += f" | Best cosine: {top[0][1][0]:.4f} | Threshold: {threshold:.3f}."
        message += " Scores are similarities, not probabilities. Threshold is exploratory unless calibrated."
        if any(r[1] == digest for r in rows):
            message += " This exact image is already registered: this is a self-match, not a generalization test."
        table, pictures = [], []
        for rank, (identity, (score, blob)) in enumerate(top, 1):
            table.append([rank, identity, round(score, 4)])
            with Image.open(io.BytesIO(blob)) as reference:
                pictures.append(
                    (reference.copy(), f"#{rank}: {identity} · cosine {score:.4f}")
                )
        return message, table, pictures

    def delete(self, identity, confirmed):
        if not confirmed:
            raise ValueError("Confirm deletion first.")
        with self.lock, self.connect() as db:
            result = db.execute("DELETE FROM images WHERE identity=?", (identity,))
            count = result.rowcount
        return f"Deleted {count} references for {identity}."


def build_app(gallery, manifest, prepared):
    """Classic dark UI on top of the current, unmodified TigerViT + SQLite system."""
    import gradio as gr
    import html
    from functools import wraps

    THRESHOLD = 0.96  # Old demo's requested default; not a calibrated operating point.
    CSS = '\nfooter { display: none !important; }\n.built-with { display: none !important; }\n#svelte-announcer { display: none !important; }\n.gradio-container > .footer { display: none !important; }\n\n@import url(\'https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap\');\n\nbody, .gradio-container {\n    font-family: \'IBM Plex Sans\', sans-serif !important;\n    background: #0e0e0e !important;\n    color: #d4d4d4 !important;\n}\n\n.header-block {\n    border-bottom: 1px solid #2a2a2a;\n    padding: 2rem 0 1.5rem 0;\n    margin-bottom: 1rem;\n}\n\n.header-block h1 {\n    font-family: \'IBM Plex Mono\', monospace !important;\n    font-size: 1.3rem !important;\n    font-weight: 500 !important;\n    letter-spacing: 0.12em !important;\n    color: #e8e8e8 !important;\n    text-transform: uppercase !important;\n    margin: 0 !important;\n}\n\n.tab-nav button {\n    font-family: \'IBM Plex Mono\', monospace !important;\n    font-size: 0.72rem !important;\n    letter-spacing: 0.1em !important;\n    text-transform: uppercase !important;\n    color: #555 !important;\n    background: transparent !important;\n    border: none !important;\n    border-bottom: 2px solid transparent !important;\n    padding: 0.6rem 1.2rem !important;\n    transition: color 0.2s, border-color 0.2s !important;\n}\n\n.tab-nav button.selected {\n    color: #d4d4d4 !important;\n    border-bottom: 2px solid #d4d4d4 !important;\n}\n\n.tab-nav button:hover { color: #999 !important; }\n\nbutton.primary {\n    font-family: \'IBM Plex Mono\', monospace !important;\n    font-size: 0.75rem !important;\n    letter-spacing: 0.1em !important;\n    text-transform: uppercase !important;\n    background: #1e1e1e !important;\n    color: #d4d4d4 !important;\n    border: 1px solid #333 !important;\n    border-radius: 2px !important;\n    padding: 0.6rem 1.4rem !important;\n    transition: background 0.2s, border-color 0.2s !important;\n    cursor: pointer !important;\n}\n\nbutton.primary:hover {\n    background: #2a2a2a !important;\n    border-color: #555 !important;\n}\n\nbutton.secondary {\n    font-family: \'IBM Plex Mono\', monospace !important;\n    font-size: 0.72rem !important;\n    letter-spacing: 0.1em !important;\n    text-transform: uppercase !important;\n    background: transparent !important;\n    color: #666 !important;\n    border: 1px solid #252525 !important;\n    border-radius: 2px !important;\n    padding: 0.6rem 1.4rem !important;\n    transition: all 0.2s !important;\n    cursor: pointer !important;\n}\n\nbutton.secondary:hover {\n    color: #aaa !important;\n    border-color: #3a3a3a !important;\n}\n\ntextarea, .gr-textbox textarea {\n    font-family: \'IBM Plex Mono\', monospace !important;\n    font-size: 0.78rem !important;\n    background: #141414 !important;\n    border: 1px solid #222 !important;\n    border-radius: 2px !important;\n    color: #b0b0b0 !important;\n    line-height: 1.7 !important;\n}\n\ninput[type="text"], .gr-textbox input {\n    font-family: \'IBM Plex Mono\', monospace !important;\n    font-size: 0.78rem !important;\n    background: #141414 !important;\n    border: 1px solid #222 !important;\n    border-radius: 2px !important;\n    color: #b0b0b0 !important;\n}\n\nlabel span, .gr-form label {\n    font-family: \'IBM Plex Mono\', monospace !important;\n    font-size: 0.68rem !important;\n    letter-spacing: 0.1em !important;\n    text-transform: uppercase !important;\n    color: #555 !important;\n}\n\n.gr-image {\n    border: 1px solid #222 !important;\n    border-radius: 2px !important;\n    background: #141414 !important;\n}\n\n.gr-gallery {\n    background: #141414 !important;\n    border: 1px solid #222 !important;\n    border-radius: 2px !important;\n}\n\n.new-tiger-panel {\n    border: 1px solid #2a2a2a;\n    border-left: 3px solid #444;\n    background: #111;\n    padding: 1rem 1.2rem 0.5rem 1.2rem;\n    margin-top: 1rem;\n    border-radius: 2px;\n}\n\n.new-tiger-label {\n    font-family: \'IBM Plex Mono\', monospace;\n    font-size: 0.68rem;\n    color: #555;\n    letter-spacing: 0.12em;\n    text-transform: uppercase;\n    margin: 0 0 0.8rem 0;\n}\n\n.gr-markdown h2, .gr-markdown h3, .gr-markdown h4 {\n    font-family: \'IBM Plex Mono\', monospace !important;\n    font-size: 0.8rem !important;\n    letter-spacing: 0.1em !important;\n    text-transform: uppercase !important;\n    color: #888 !important;\n    font-weight: 500 !important;\n    border-bottom: 1px solid #1e1e1e !important;\n    padding-bottom: 0.4rem !important;\n    margin-top: 1.4rem !important;\n}\n\n.gr-markdown p, .gr-markdown li {\n    font-size: 0.82rem !important;\n    color: #666 !important;\n    line-height: 1.8 !important;\n}\n\n.gr-markdown code, .gr-markdown pre {\n    font-family: \'IBM Plex Mono\', monospace !important;\n    font-size: 0.74rem !important;\n    background: #141414 !important;\n    border: 1px solid #222 !important;\n    color: #888 !important;\n    border-radius: 2px !important;\n}\n\n.threshold-note {\n    font-family: \'IBM Plex Mono\', monospace;\n    font-size: 0.68rem;\n    color: #444;\n    letter-spacing: 0.08em;\n    margin: 0.3rem 0 1rem 0;\n}\n\n.divider {\n    border: none;\n    border-top: 1px solid #1e1e1e;\n    margin: 1rem 0;\n}\n\n.gradio-container { max-width: 1120px !important; margin: auto !important; }\n.gradio-container, .gradio-container * { box-sizing: border-box; }\n.gradio-container { --body-background-fill: #0e0e0e; --block-background-fill: #141414; --block-border-color: #2a2a2a; --input-background-fill: #141414; --body-text-color: #d4d4d4; --block-label-text-color: #a2a2a2; --button-primary-background-fill: #282828; --button-primary-background-fill-hover: #373737; --button-primary-text-color: #f0f0f0; --color-accent-soft: #282828; }\n.gradio-container .block, .gradio-container .form { background: #141414 !important; border-color: #292929 !important; border-radius: 4px !important; }\n.gradio-container input, .gradio-container textarea { color: #ddd !important; }\n.gradio-container .tab-nav { border-color: #2a2a2a !important; }\n.gradio-container button { border-radius: 3px !important; }\n.gradio-container button:focus-visible { outline: 2px solid #aaa !important; outline-offset: 2px; }\n.gradio-container table { color: #ccc !important; }\n.gradio-container h1, .gradio-container h2, .gradio-container h3 { color: #e8e8e8 !important; }\n'

    def guarded(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except (ValueError, OSError, RuntimeError) as error:
                raise gr.Error(str(error)) from error
        return wrapped

    def inventory_update():
        rows = gallery.inventory()
        return rows, gr.update(choices=[r[0] for r in rows], value=None)

    def stats_text():
        rows = gallery.inventory()
        return (f"Existing database ready.\n\nIndividuals : {len(rows)}\n"
                f"Total images: {sum(n for _, n in rows)}\n\nKnown tigers:\n"
                + "\n".join(f"  {name}" for name, _ in rows)) if rows else (
                    "Gallery is empty. Use Build Database from Images to import references."
                )

    def load_existing():
        return stats_text(), *inventory_update()

    def build_database(source, excluded, progress=gr.Progress()):
        # Underlying new project imports from a validated manifest, not a free-form folder.
        result = gallery.import_manifest(manifest, prepared, source, excluded, progress)
        return result + "\n\n" + stats_text(), *inventory_update()

    def reference_images(identity, limit=24):
        with gallery.lock, gallery.connect() as db:
            blobs = db.execute(
                "SELECT image FROM images WHERE identity=? ORDER BY id LIMIT ?",
                (identity, limit),
            ).fetchall()
        result = []
        for (blob,) in blobs:
            with Image.open(io.BytesIO(blob)) as im:
                result.append((im.copy(), identity))
        return result

    def identify_old_style(value):
        message, table, top_images = gallery.identify(value, THRESHOLD)
        if not table:
            return message, [], []
        best_id = str(table[0][1])
        best_score = float(table[0][2])
        known = best_score >= THRESHOLD
        lines = ["RESULT — MATCH FOUND" if known else "RESULT — NEW TIGER DETECTED",
                 "─" * 40, ""]
        if known:
            lines.extend([f"Identity   : {best_id}", f"Similarity : {best_score:.2%}"])
        else:
            lines.extend(["No match met the similarity threshold.",
                          f"Threshold  : {THRESHOLD:.0%}",
                          f"Best score : {best_score:.2%}"])
        lines.extend(["", "Top similarities:"])
        for rank, name, score in table:
            lines.append(f"  {rank}. {name}  {float(score):.2%}")
        lines.extend(["", "Cosine similarity is not a probability; 0.96 is not calibrated."])
        if "self-match" in message:
            lines.append("Warning: exact reference image already registered (self-match).")
        # Mimic the old demo: show references only for the accepted identity.
        refs = reference_images(best_id) if known else []
        return "\n".join(lines), refs, top_images

    def save_reference(value, identity, reviewed):
        identity = gallery.valid_id(identity)
        known_ids = {r[0] for r in gallery.inventory()}
        mode = "Existing tiger" if identity in known_ids else "New tiger"
        result = gallery.enroll(value, identity, mode, reviewed)
        return result, *inventory_update(), False

    def remove_identity(identity, reviewed):
        result = gallery.delete(identity, reviewed)
        return result, *inventory_update(), False

    def batch_identify(directory):
        # Same manual batch workflow as the original demo, using the current encoder.
        directory = str(directory or "data/test").strip()
        root = Path(directory).expanduser()
        if not root.is_absolute():
            root = Path.cwd() / root
        root = root.resolve()
        if not root.is_dir():
            raise ValueError(f"Batch directory not found: {root}")
        if not gallery.inventory():
            raise ValueError("Gallery is empty; import references first.")
        paths = sorted(p for p in root.rglob("*") if p.is_file() and
                       p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
        if not paths:
            return f"No images found in {root}"
        lines = [f"BATCH RESULTS — {len(paths)} images", "─" * 45, ""]
        for file in paths:
            try:
                _, rows, _ = gallery.identify(file, THRESHOLD)
                if not rows:
                    lines.append(f"{file.name}: no matches")
                    continue
                best_id, score = str(rows[0][1]), float(rows[0][2])
                label = best_id if score >= THRESHOLD else "NEW TIGER"
                lines.extend([str(file.relative_to(root)), f"  → {label}  ({score:.2%})", ""])
            except (ValueError, OSError, RuntimeError) as error:
                lines.extend([str(file.relative_to(root)), f"  → ERROR: {error}", ""])
        return "\n".join(lines)

    theme = gr.themes.Base(
        primary_hue="neutral", secondary_hue="neutral", neutral_hue="neutral",
        font=["IBM Plex Sans", "Arial", "sans-serif"],
        font_mono=["IBM Plex Mono", "monospace"],
    )
    with gr.Blocks(title="Tiger Individual Identification System") as app:
        gr.HTML('<div class="header-block"><h1>Tiger Individual Identification System</h1></div>')
        with gr.Tab("Setup"):
            gr.Markdown("#### Database")
            gr.Markdown("Load the existing SQLite gallery, or import reference images from the experiment manifest. The trained model is unchanged.")
            with gr.Row():
                load_btn = gr.Button("Load Existing Database", variant="primary")
                build_btn = gr.Button("Build Database from Images", variant="secondary")
            with gr.Row():
                source = gr.Radio(["test-gallery", "train"], value="test-gallery",
                                  label="Reference source")
                excluded = gr.Textbox(label="IDs to withhold (optional)",
                                      placeholder="Comma-separated exact tiger IDs")
            setup_status = gr.Textbox(label="Status", value=stats_text(), lines=10,
                                      interactive=False)
            gr.Markdown("#### Delete a tiger")
            with gr.Row():
                deletion = gr.Dropdown(choices=[r[0] for r in gallery.inventory()],
                                       label="Tiger name / ID to delete", scale=3)
                delete_btn = gr.Button("Delete Tiger", variant="secondary", scale=1)
            delete_confirm = gr.Checkbox(label="I confirm removal of this tiger and all its gallery references.")
            delete_status = gr.Textbox(label="Deletion result", interactive=False)
            inventory = gr.Dataframe(value=gallery.inventory(),
                                     headers=["Tiger ID", "References"], interactive=False)
            load_btn.click(guarded(load_existing), outputs=[setup_status, inventory, deletion])
            build_btn.click(guarded(build_database), [source, excluded],
                            [setup_status, inventory, deletion])
            delete_btn.click(guarded(remove_identity), [deletion, delete_confirm],
                             [delete_status, inventory, deletion, delete_confirm])
            deletion.change(lambda: False, outputs=delete_confirm)

        with gr.Tab("🔍 Identify"):
            gr.Markdown("#### Single Image Identification")
            gr.HTML('<p class="threshold-note">Similarity threshold: 96% &nbsp;·&nbsp; '
                    'Below threshold → New Tiger Detected &nbsp;·&nbsp; '
                    'This is an illustrative threshold, not a calibrated confidence.</p>')
            with gr.Row():
                with gr.Column(scale=1):
                    query = gr.Image(label="Query Image", type="pil")
                    identify_btn = gr.Button("Identify", variant="primary")
                with gr.Column(scale=1):
                    result = gr.Textbox(label="Result", lines=14, interactive=False)
            gr.Markdown("#### Reference images of matched individual")
            references = gr.Gallery(label="Reference images", columns=4, height=300)
            with gr.Accordion("Top 3 candidate matches", open=False):
                candidates = gr.Gallery(label="Best reference for each candidate", columns=3)
            gr.Markdown("#### Register a new reference")
            gr.HTML('<p class="threshold-note">Review the image and matches before saving. '
                    'Existing IDs receive an additional reference; other IDs create a new tiger.</p>')
            with gr.Row():
                tiger_name = gr.Textbox(label="Tiger name / ID", placeholder="e.g. tiger_275", scale=3)
                save_btn = gr.Button("Add Tiger to Database", variant="secondary", scale=1)
            save_confirm = gr.Checkbox(label="I reviewed the photo and confirm this identity.")
            save_status = gr.Textbox(label="Enrollment result", interactive=False)
            identify_btn.click(guarded(identify_old_style), [query],
                               [result, references, candidates])
            query.change(lambda: ("", [], [], False),
                         outputs=[result, references, candidates, save_confirm])
            tiger_name.change(lambda: False, outputs=save_confirm)
            save_btn.click(guarded(save_reference), [query, tiger_name, save_confirm],
                           [save_status, inventory, deletion, save_confirm])

        with gr.Tab("Batch"):
            gr.Markdown("#### Batch Identification")
            gr.Markdown("Process images from a local directory against the current SQLite gallery.")
            batch_dir = gr.Textbox(label="Query directory", value="data/test")
            batch_btn = gr.Button("Run Batch", variant="primary")
            batch_output = gr.Textbox(label="Batch Results", lines=20, interactive=False)
            batch_btn.click(guarded(batch_identify), [batch_dir], [batch_output])

        with gr.Tab("ℹ️ Info"):
            gr.Markdown("""#### About
This demo retrieves individual tiger identities using the **new project's trained TigerViT**.
No model layers, checkpoints, training loss functions, or preprocessing have been changed.

#### How it works
- **Embedding** — 128-dimensional L2-normalized vector produced by the current model
- **Matching** — maximum cosine similarity over each identity's stored references
- **Database** — local SQLite gallery, compatible with the new project's manifest importer
- **Threshold** — 0.96 (requested old-demo setting; not empirically calibrated for this checkpoint)
- **Unknown tiger** — no registered identity reaches the threshold

#### Research caution
Demo identification is different from held-out evaluation. Reference/query self-matches
and manually added images should not be counted as independent test performance.
""")
    app._classic_theme = theme
    app._classic_css = CSS
    return app.queue(default_concurrency_limit=1)

def main(project_root=None):
    base = Path(project_root or Path.cwd()).resolve()
    parser = argparse.ArgumentParser(
        description="Local Gradio tiger gallery demo (no training)."
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to an existing run's best.pt"
    )
    parser.add_argument("--database", default="artifacts/demo/gallery.sqlite3")
    parser.add_argument("--manifest", default="artifacts/experiment/manifest.json")
    parser.add_argument("--prepared-root", default="artifacts/prepared/train")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    def resolve(value):
        return (base / value).resolve()

    try:
        import gradio  # noqa: F401
    except ImportError:
        parser.exit(
            1,
            'Install the optional demo dependency: python -m pip install "gradio>=6,<7"\n',
        )
    try:
        encoder = Encoder(resolve(args.checkpoint), args.device)
        gallery = Gallery(resolve(args.database), encoder)
        app = build_app(gallery, resolve(args.manifest), resolve(args.prepared_root))
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Demo startup failed: {error}\n")
    # Local only; return PIL images rather than exposing dataset/checkpoint directories.
    app.launch(
        server_name="127.0.0.1", server_port=args.port, share=False, inbrowser=True,
        theme=app._classic_theme, css=app._classic_css,
    )


if __name__ == "__main__":
    main()
