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
    import gradio as gr

    def guarded(fn):
        from functools import wraps

        @wraps(fn)
        def wrapped(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except (ValueError, OSError, RuntimeError) as error:
                raise gr.Error(str(error)) from error

        return wrapped

    def refresh():
        rows = gallery.inventory()
        ids = [r[0] for r in rows]
        return rows, gr.Dropdown(choices=ids, value=None)

    with gr.Blocks(title="Tiger Re-ID · Gallery Demo") as app:
        gr.Markdown(
            "# Tiger Re-ID\nUpload a tiger photograph, inspect its closest known identities, and manage your local gallery."
        )
        gr.Markdown(
            f"Model: `{gallery.encoder.config.get('loss', 'trained ViT')}` · "
            f"seed `{gallery.encoder.config.get('seed', '?')}` · 128-dimensional embeddings. "
            "Gallery changes do not retrain the model or alter research results."
        )
        with gr.Tab("Identify & register"):
            with gr.Row():
                query = gr.Image(type="pil", label="Query tiger photograph")
                with gr.Column():
                    threshold = gr.Slider(
                        -1,
                        1,
                        value=0.6,
                        step=0.005,
                        label="Cosine threshold (uncalibrated demo default)",
                    )
                    identify = gr.Button("Find top 3 tigers", variant="primary")
                    status = gr.Textbox(label="Decision", interactive=False, lines=4)
            matches = gr.Gallery(
                label="Best reference for each distinct identity", columns=3
            )
            scores = gr.Dataframe(
                headers=["Rank", "Tiger ID", "Cosine similarity"], interactive=False
            )
            gr.Markdown(
                "### Register this photograph\nReview the candidates first. A low score can also mean a difficult pose or poor image quality. New IDs require human confirmation."
            )
            mode = gr.Radio(
                ["New tiger", "Existing tiger"], value="New tiger", label="Save as"
            )
            identity = gr.Textbox(
                label="Tiger ID",
                placeholder="e.g. new_tiger_001; for an existing tiger, copy its ID from Manage gallery",
            )
            confirm = gr.Checkbox(
                label="I reviewed the photograph and matches, and confirm this identity."
            )
            save = gr.Button("Save reference to gallery")
            saved = gr.Textbox(label="Enrollment result", interactive=False)
            identify.click(
                guarded(gallery.identify), [query, threshold], [status, scores, matches]
            )
            query.change(
                lambda: ("", [], [], False), outputs=[status, scores, matches, confirm]
            )
            threshold.change(
                lambda: ("Threshold changed; run Find top 3 again.", [], []),
                outputs=[status, scores, matches],
            )
            mode.change(lambda: False, outputs=confirm)
            identity.change(lambda: False, outputs=confirm)
        with gr.Tab("Manage gallery"):
            gr.Markdown(
                "Import references from the existing split manifest. **train** imports training images; **test-gallery** imports only held-out gallery references, never query images. Imports add to the current database."
            )
            source = gr.Radio(
                ["train", "test-gallery"], value="train", label="Reference source"
            )
            excluded = gr.Textbox(
                label="IDs to withhold (comma separated, exact manifest IDs)",
                placeholder="Leave blank to register all selected identities",
            )
            load = gr.Button("Import reference images", variant="primary")
            import_status = gr.Textbox(label="Import result", interactive=False)
            inventory = gr.Dataframe(
                value=gallery.inventory(),
                headers=["Tiger ID", "References"],
                interactive=False,
            )
            refresh_button = gr.Button("Refresh inventory")
            deletion = gr.Dropdown(
                choices=[r[0] for r in gallery.inventory()], label="Tiger to remove"
            )
            delete_confirm = gr.Checkbox(
                label="Remove this identity and all its reference images from the demo database."
            )
            delete_button = gr.Button("Delete selected identity")
            deleted = gr.Textbox(label="Deletion result", interactive=False)

            def do_import(selection, omit, progress=gr.Progress()):
                try:
                    result = gallery.import_manifest(
                        manifest, prepared, selection, omit, progress
                    )
                    return result, *refresh()
                except (ValueError, OSError, RuntimeError) as error:
                    raise gr.Error(str(error)) from error

            def do_save(value, tiger_id, selected_mode, reviewed):
                result = gallery.enroll(value, tiger_id, selected_mode, reviewed)
                return result, *refresh(), False

            def do_delete(tiger_id, reviewed):
                result = gallery.delete(tiger_id, reviewed)
                return result, *refresh(), False

            load.click(
                do_import, [source, excluded], [import_status, inventory, deletion]
            )
            save.click(
                guarded(do_save),
                [query, identity, mode, confirm],
                [saved, inventory, deletion, confirm],
            )
            refresh_button.click(refresh, outputs=[inventory, deletion])
            delete_button.click(
                guarded(do_delete),
                [deletion, delete_confirm],
                [deleted, inventory, deletion, delete_confirm],
            )
            deletion.change(lambda: False, outputs=delete_confirm)
        with gr.Tab("How the demo works"):
            gr.Markdown("""### Retrieval, not automatic species recognition
The same ViT and evaluation preprocessing produce normalized 128-dimensional embeddings.
Each identity gets the **maximum cosine similarity** across its references. We show up to
three distinct identities. The top identity is a candidate only when its score is at least
the threshold. The app expects tiger crops; it does not detect animals or validate species.

### Demonstrate an unknown animal
Use a fresh database and import **test-gallery**, withholding one or more exact IDs.
Upload a separate query photograph of a registered identity. Then try a query of a withheld
identity, inspect the warning, and manually register it. Use a *different photograph* to
query that newly registered tiger again. Unknowns can still falsely match a known animal.

### Threshold and research boundaries
0.6 is an illustrative default, not a validated operating point. Calibrate using validation
identities split into known and withheld groups; measure false accepts and false rejects.
Do not tune against final-test results. Changing the gallery or model can change the suitable
threshold. This mutable demo is separate from the fixed statistical evaluation.
""")
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
        server_name="127.0.0.1", server_port=args.port, share=False, inbrowser=True
    )


if __name__ == "__main__":
    main()
