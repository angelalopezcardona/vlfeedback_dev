from typing import List, Dict, Tuple, Optional
import math
import numpy as np
import os
from typing import Dict, Tuple
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from typing import Dict, Tuple, Optional
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import cm

class ImageProcessor:
    
    @staticmethod
    def draw_all_patches_and_save(
        image_path: str,
        info: Dict[str, object],          # returned by map_single_image_tokens(...)
        save_folder: str,
        image_name: str,
        edgecolor: str = "orange",
        linewidth: float = 0.6,
        annotate: bool = False,
        fontsize: int = 6,
        dpi: int = 150,
    ) -> str:
        """
        Draws ALL patch boxes (no fill) on the image resized to the processed size and saves it.

        Parameters
        ----------
        image_path : path to the original image
        info       : dict from map_single_image_tokens(...)
                    expects keys: 'processed_size' -> (H, W), 'grid' -> (gh, gw), 'patch_size' -> p
        save_folder: output directory
        image_name : filename to save ('.png' will be added if no extension)
        edgecolor  : box outline color
        linewidth  : box outline width
        annotate   : write (row,col) in each patch
        fontsize   : font size for annotations
        dpi        : output dpi

        Returns
        -------
        save_path  : full path to the saved image
        """
        # Ensure output path
        os.makedirs(save_folder, exist_ok=True)
        root, ext = os.path.splitext(image_name)
        if ext == "":
            image_name = root + ".png"
        save_path = os.path.join(save_folder, image_name)

        # Read and resize image to processed size (so patches align perfectly)
        H, W = info["processed_size"]           # (height, width) used by the vision tower
        gh, gw = info["grid"]                   # grid rows/cols
        p = info["patch_size"]                  # patch size in pixels
        img = Image.open(image_path).convert("RGB").resize((W, H), resample=Image.BILINEAR)

        # Create figure sized roughly to image pixels
        fig_w, fig_h = W / 100.0, H / 100.0
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
        ax.imshow(img)
        ax.set_axis_off()

        # Draw ALL patches (row-major)
        for r in range(gh):
            for c in range(gw):
                left  = c * p
                top   = r * p
                rect = Rectangle(
                    (left, top),
                    p, p,
                    linewidth=linewidth,
                    edgecolor=edgecolor,
                    facecolor="none"
                )
                ax.add_patch(rect)
                if annotate:
                    ax.text(
                        left + 2, top + 10,
                        f"{r},{c}",
                        fontsize=fontsize,
                        color=edgecolor,
                        ha="left", va="top",
                        bbox=dict(facecolor="black", alpha=0.2, pad=0.3)
                    )

        plt.tight_layout(pad=0)
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        return save_path

    @staticmethod
    def _final_hw_from_processor(ip) -> Tuple[int, int]:
        """Infer processed (H, W) used by the vision tower from a CLIPImageProcessor."""
        if getattr(ip, "do_center_crop", False) and hasattr(ip, "crop_size"):
            H = int(ip.crop_size.get("height", 0))
            W = int(ip.crop_size.get("width", 0))
            if H and W:
                return H, W
        # Fallback (square shortest_edge)
        if hasattr(ip, "size") and isinstance(ip.size, dict):
            s = int(ip.size.get("shortest_edge", 0))
            if s:
                return s, s
        return 336, 336  # safe default for your printout

    @staticmethod
    def _image_token_id(tokenizer) -> int:
        tid = tokenizer.convert_tokens_to_ids("<image>")
        if tid is None or tid == tokenizer.unk_token_id:
            raise ValueError("Could not find '<image>' token id in tokenizer.")
        return tid

    @staticmethod
    def _find_single_block(ids: List[int], image_tid: int) -> Tuple[int, int]:
        """Return (start, length) of the ONLY <image> block; raises if 0 or >1 blocks."""
        n = len(ids)
        blocks = []
        i = 0
        while i < n:
            if ids[i] == image_tid:
                j = i
                while j < n and ids[j] == image_tid:
                    j += 1
                blocks.append((i, j - i))
                i = j
            else:
                i += 1
        if len(blocks) == 0:
            raise ValueError("No <image> block found in input_ids.")
        if len(blocks) > 1:
            raise ValueError("Multiple <image> blocks found; this helper assumes exactly one image.")
        return blocks[0]

    @staticmethod
    def _infer_grid_and_cls(H: int, W: int, block_len: int,
                            assume_patch_size: Optional[int] = 14) -> Tuple[int, int, int, bool]:
        """
        Returns (patch_size, grid_h, grid_w, has_cls).
        Uses common ViT patch sizes; tolerates +1 CLS token.
        """
        # Try a provided/assumed patch size first (LLaVA often uses ViT-L/14).
        if assume_patch_size:
            p = assume_patch_size
            if H % p == 0 and W % p == 0:
                gh, gw = H // p, W // p
                expected = gh * gw
                if block_len == expected:
                    return p, gh, gw, False
                if block_len == expected + 1:
                    return p, gh, gw, True
        # Try common patch sizes
        for p in (8, 14, 16, 32):
            if H % p == 0 and W % p == 0:
                gh, gw = H // p, W // p
                expected = gh * gw
                if block_len == expected:
                    return p, gh, gw, False
                if block_len == expected + 1:
                    return p, gh, gw, True
        # Last resort: factor near-square with/without +1
        s = int(round(math.sqrt(max(1, block_len - 1))))
        for gh in range(max(1, s - 5), s + 6):
            if (block_len % gh) == 0:
                gw = block_len // gh
                if (H % gh == 0) and (W % gw == 0) and (H // gh == W // gw):
                    return H // gh, gh, gw, False
            if ((block_len - 1) > 0) and ((block_len - 1) % gh == 0):
                gw = (block_len - 1) // gh
                if (H % gh == 0) and (W % gw == 0) and (H // gh == W // gw):
                    return H // gh, gh, gw, True
        raise ValueError(f"Cannot infer patch grid for H={H}, W={W}, block_len={block_len}")


    def map_single_image_tokens(self,
        input_ids: List[int],
        tokenizer,
        image_processor,
        assume_patch_size: Optional[int] = 14
    ) -> Dict[str, object]:
        """
        Returns a dict with:
        - 'start': int, start index of the image block in input_ids
        - 'length': int, total tokens in the block (including CLS if present)
        - 'has_cls': bool, whether the block includes a visual CLS (assumed at start)
        - 'grid': (gh, gw)
        - 'patch_size': int, patch side in pixels
        - 'index_to_patch': dict {seq_idx -> (r, c) or None for CLS}
        - 'index_to_bbox': dict {seq_idx -> (left, top, right, bottom) or None for CLS} in processed-image pixels
        - 'patch_to_seq': np.ndarray shape (gh, gw) with sequence indices (CLS not included)
        """
        H, W = self._final_hw_from_processor(image_processor)
        img_tid = self._image_token_id(tokenizer)
        start, length = self._find_single_block(input_ids, img_tid)
        p, gh, gw, has_cls = self._infer_grid_and_cls(H, W, length, assume_patch_size)

        index_to_patch: Dict[int, Optional[Tuple[int, int]]] = {}
        index_to_bbox: Dict[int, Optional[Tuple[int, int, int, int]]] = {}
        patch_to_seq = np.full((gh, gw), -1, dtype=int)

        for idx in range(start, start + length):
            rel = idx - start
            if has_cls and rel == 0:
                # CLS at start (typical)
                index_to_patch[idx] = None
                index_to_bbox[idx] = None
                continue

            rel_eff = rel - (1 if has_cls else 0)
            r = rel_eff // gw
            c = rel_eff % gw
            # safety guard
            if not (0 <= r < gh and 0 <= c < gw):
                index_to_patch[idx] = None
                index_to_bbox[idx] = None
                continue

            top = r * p
            left = c * p
            bottom = (r + 1) * p
            right = (c + 1) * p

            index_to_patch[idx] = (r, c)
            index_to_bbox[idx] = (left, top, right, bottom)
            patch_to_seq[r, c] = idx

        return {
            "start": start,
            "length": length,
            "has_cls": has_cls,
            "grid": (gh, gw),
            "patch_size": p,
            "index_to_patch": index_to_patch,
            "index_to_bbox": index_to_bbox,
            "patch_to_seq": patch_to_seq,
            "processed_size": (H, W),
        }

    @staticmethod
    def sequence_attention_to_patch_heatmap(
        seq_attn: np.ndarray,
        start: int,
        gh: int,
        gw: int,
        has_cls: bool
    ) -> np.ndarray:
        """
        Convert a 1D attention vector over the *whole sequence* into a (gh, gw) heatmap
        for the single image block.
        seq_attn: shape (seq_len,)
        """
        offset = 1 if has_cls else 0
        block = seq_attn[start + offset : start + offset + gh * gw]
        if isinstance(block, list):
            block = np.array(block)
        if block.size != gh * gw:
            raise ValueError("Attention segment does not match grid size.")
        heat = block.reshape(gh, gw)
        # normalize to [0, 1] (optional)
        denom = (heat.max() - heat.min()) or 1.0
        return (heat - heat.min()) / denom
    
    @staticmethod
    def save_attention_overlay(
        image_path: str,
        heat: np.ndarray,              # (gh, gw) from sequence_attention_to_patch_heatmap(...)
        info: Dict[str, object],       # dict from map_single_image_tokens(...)
        save_folder: str,
        image_name: str,
        cmap_name: str = "viridis",    # colorblind-friendly
        alpha: float = 0.45,           # overlay strength (0..1)
        dpi: int = 200,                # only affects metadata for PNG viewers
    ) -> str:
        """
        Overlay a (gh, gw) attention heatmap on the image and save it.

        - Resizes the original image to info['processed_size'] so grid aligns.
        - Upsamples heat to the same size, colorizes, and blends with alpha.

        Returns the full output path.
        """
        # --- prepare output path ---
        os.makedirs(save_folder, exist_ok=True)
        root, ext = os.path.splitext(image_name)
        if ext == "":
            image_name = root + ".png"
        out_path = os.path.join(save_folder, image_name)

        # --- load & resize image to processed size ---
        H, W = info["processed_size"]            # (height, width)
        gh, gw = info["grid"]
        if heat.shape != (gh, gw):
            raise ValueError(f"heat shape {heat.shape} != grid {(gh, gw)}")

        img = Image.open(image_path).convert("RGB").resize((W, H), resample=Image.BILINEAR)
        img_np = np.array(img).astype(np.float32)

        # --- normalize heat to [0,1] (robust to constants) ---
        h = heat.astype(np.float32)
        h_min, h_max = float(np.nanmin(h)), float(np.nanmax(h))
        if h_max - h_min > 1e-12:
            h = (h - h_min) / (h_max - h_min)
        else:
            h = np.zeros_like(h)

        # --- upsample heat to image size ---
        h_img = Image.fromarray((h * 255).astype(np.uint8)).resize((W, H), resample=Image.BILINEAR)
        h_arr = np.asarray(h_img).astype(np.float32) / 255.0  # [0,1]

        # --- colorize heat ---
        from matplotlib import cm
        cmap = cm.get_cmap(cmap_name)
        heat_rgba = (cmap(h_arr) * 255).astype(np.uint8)      # HxWx4
        heat_rgb = heat_rgba[..., :3].astype(np.float32)

        # --- blend overlay ---
        overlay = (1.0 - alpha) * img_np + alpha * heat_rgb
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        Image.fromarray(overlay).save(out_path, dpi=(dpi, dpi))
        return out_path


    def save_attention_overlay_with_grid(
        self,
        image_path: str,
        heat: np.ndarray,                    # (gh, gw) attention over patches
        info: Dict[str, object],             # expects keys: 'processed_size' -> (H, W), 'grid' -> (gh, gw), 'patch_size' -> p
        save_folder: str,
        image_name: str,
        # overlay options
        cmap_name: str = "viridis",
        alpha: float = 0.45,
        dpi: int = 200,
        # grid options
        draw_all_patches: bool = True,
        grid_edgecolor: str = "orange",
        grid_linewidth: float = 0.6,
        annotate: bool = False,
        fontsize: int = 6,
        # top-k options
        draw_topk: bool = True,
        topk: int = 10,
        topk_color: str = "red",
        topk_linewidth: float = 2.0,
        # extra debug
        save_boxes_only: bool = True,
    ) -> Dict[str, str]:
        """
        Creates:
        - overlay image with heat + optional full grid + optional Top-K boxes
        - (optional) boxes-only debug over resized original

        Returns dict with saved paths.
        """
        os.makedirs(save_folder, exist_ok=True)
        root, ext = os.path.splitext(image_name)
        if ext == "":
            image_name = root + ".png"
        overlay_path = os.path.join(save_folder, image_name)
        boxes_only_path = os.path.join(save_folder, f"{root}_topk_boxes.png")

        # --- sizes & checks ---
        H, W = info["processed_size"]          # (height, width)
        gh, gw = info["grid"]
        p = info["patch_size"]
        if heat.shape != (gh, gw):
            raise ValueError(f"heat shape {heat.shape} != grid {(gh, gw)}")

        # --- load & resize base image ---
        img = Image.open(image_path).convert("RGB").resize((W, H), resample=Image.BILINEAR)
        img_np = np.array(img).astype(np.float32)  # HxWx3

        # --- normalize heat to [0,1] ---
        h = heat.astype(np.float32)
        h_min, h_max = float(np.nanmin(h)), float(np.nanmax(h))
        h = (h - h_min) / (h_max - h_min) if (h_max - h_min) > 1e-12 else np.zeros_like(h, dtype=np.float32)

        # --- upsample heat to image size (stay float) ---
        h_up = np.array(Image.fromarray(h).resize((W, H), resample=Image.BILINEAR), dtype=np.float32)
        h_up = np.clip(h_up, 0.0, 1.0)

        # --- colorize + blend ---
        heat_rgb = (cm.get_cmap(cmap_name)(h_up)[..., :3] * 255.0).astype(np.float32)
        overlay_np = np.clip((1.0 - alpha) * img_np + alpha * heat_rgb, 0, 255).astype(np.uint8)

        # helper to draw grid & top-k using matplotlib (cleanest for annotations)
        def draw_fig(base_rgb: np.ndarray, draw_grid: bool, draw_k: bool, save_path: str):
            fig_w, fig_h = W / 100.0, H / 100.0
            fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
            ax.imshow(base_rgb)
            ax.set_axis_off()

            # Grid of all patches (row-major). Uses `p` for rectangle size.
            if draw_grid:
                for r in range(gh):
                    for c in range(gw):
                        left = c * p
                        top  = r * p
                        rect = Rectangle(
                            (left, top), p, p,
                            linewidth=grid_linewidth,
                            edgecolor=grid_edgecolor,
                            facecolor="none"
                        )
                        ax.add_patch(rect)
                        if annotate:
                            ax.text(
                                left + 2, top + 10,
                                f"{r},{c}",
                                fontsize=fontsize,
                                color=grid_edgecolor,
                                ha="left", va="top",
                                bbox=dict(facecolor="black", alpha=0.2, pad=0.3)
                            )

            # Top-K boxes based on heat grid (robust to non-multiple sizes via linspace)
            if draw_k and topk > 0:
                flat = h.ravel()
                k = min(topk, flat.size)
                idxs = np.argsort(flat)[::-1][:k]
                y_edges = np.linspace(0, H, gh + 1, dtype=int)
                x_edges = np.linspace(0, W, gw + 1, dtype=int)
                for idx in idxs:
                    r, c = divmod(idx, gw)
                    x0, x1 = x_edges[c], x_edges[c + 1]
                    y0, y1 = y_edges[r], y_edges[r + 1]
                    rect = Rectangle(
                        (x0, y0), x1 - x0, y1 - y0,
                        linewidth=topk_linewidth, edgecolor=topk_color, facecolor="none"
                    )
                    ax.add_patch(rect)

            plt.tight_layout(pad=0)
            fig.savefig(save_path, bbox_inches="tight", pad_inches=0)
            plt.close(fig)

        # 1) Save overlay (with grid + top-k if requested)
        draw_fig(overlay_np, draw_grid=draw_all_patches, draw_k=draw_topk, save_path=overlay_path)

        # 2) Optional: boxes-only debug over the resized original
        if save_boxes_only:
            draw_fig(img_np.astype(np.uint8), draw_grid=draw_all_patches, draw_k=draw_topk, save_path=boxes_only_path)

        out = {"overlay": overlay_path}
        if save_boxes_only:
            out["boxes_only"] = boxes_only_path
        return out

