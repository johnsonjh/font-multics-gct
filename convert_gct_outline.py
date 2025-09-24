#!/usr/bin/env python3
# SPDX-License-Identifier: Multics or MIT-0
# Copyright (c) 2025 Jeffrey H. Johnson
# Copyright (c) 2025 The DPS8M Development Team
# scspell-id: 5db54df4-9924-11f0-b947-80ee73e9b8e7

import fontforge
import os
import re
import argparse
import multiprocessing
import tempfile
import shutil
from functools import partial

SCALE = 20
Y_OFFSET = 380
STROKE_WIDTH = 15


def gct_name_to_font_name(gct_name):
    base_name = gct_name.replace("gct_", "").strip("_")
    parts = base_name.split("_")
    return "GCT" + "".join([p.capitalize() for p in parts])


def gct_name_to_family_name(gct_name):
    base_name = gct_name.replace("gct_", "").strip("_")
    parts = base_name.split("_")
    return "GCT " + " ".join([p.capitalize() for p in parts])


def get_glyph_chunks(lines):
    chunks = []
    current_chunk = []
    for line in lines:
        stripped = line.strip()
        if stripped and stripped.endswith(":") and not stripped.startswith("metric"):
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = [line]
        elif current_chunk:
            current_chunk.append(line)
    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def process_glyph_chunk(glyph_lines, tmpdir, name_map):
    glyph_name = glyph_lines[0].strip()[:-1]
    std_name = name_map.get(glyph_name, glyph_name)

    try:
        font = fontforge.font()
        font.em = 1000
        current_glyph = font.createChar(-1, std_name)
        print(f"Processing glyph: {glyph_name} -> {std_name}", flush=True)

        x, y = 0, 0
        all_strokes = []
        current_stroke = []

        for line in glyph_lines[1:]:
            line = line.strip()
            if not line or line.startswith("metric"):
                continue

            parts = re.split(r"\s+", line)
            command = parts[0]

            if command == "shift" or command == "end":
                if current_stroke:
                    all_strokes.append(current_stroke)
                current_stroke = []

                if command == "shift":
                    try:
                        dx, dy = int(parts[1]), int(parts[2])
                        x += dx
                        y += dy
                        current_stroke.append((x, y))
                    except (ValueError, IndexError):
                        print(
                            f"Warning: Could not parse coordinates in {font_name_base} shift: {line}",
                            flush=True,
                        )
                else:  # end
                    if current_glyph:
                        for stroke_points in all_strokes:
                            if not stroke_points or len(stroke_points) < 2:
                                continue
                            scaled_points = [
                                (p[0] * SCALE, p[1] * SCALE + Y_OFFSET)
                                for p in stroke_points
                            ]
                            contour = fontforge.contour()
                            contour.moveTo(*scaled_points[0])
                            for point in scaled_points[1:]:
                                contour.lineTo(*point)
                            current_glyph.layers[1] += contour

                        current_glyph.width = x * SCALE
                        current_glyph.stroke("circular", STROKE_WIDTH)
                        current_glyph.removeOverlap()
                        current_glyph.correctDirection()

            elif command == "vector":
                if not current_stroke:
                    current_stroke.append((x, y))
                try:
                    dx, dy = int(parts[1]), int(parts[2])
                    x += dx
                    y += dy
                    current_stroke.append((x, y))
                except (ValueError, IndexError):
                    print(
                        f"Warning: Could not parse coordinates in {font_name_base} vector: {line}",
                        flush=True,
                    )

        glyph_sfd_path = os.path.join(tmpdir, f"{std_name}.sfd")
        font.save(glyph_sfd_path)
        return glyph_sfd_path
    except Exception as e:
        print(f"Error processing glyph {glyph_name}: {e}", flush=True)
        return None


def convert_gct_to_sfd(input_path, output_path):
    font_name_base = os.path.basename(input_path)
    font_name = gct_name_to_font_name(font_name_base)
    family_name = gct_name_to_family_name(font_name_base)

    name_map = {
        "bel": "bell",
        "excl_pt": "exclam",
        "dbl_quot": "quotedbl",
        "sharp": "numbersign",
        "dollar": "dollar",
        "percent": "percent",
        "amprsnd": "ampersand",
        "r_quote": "quoteright",
        "l_paren": "parenleft",
        "r_paren": "parenright",
        "star": "asterisk",
        "plus": "plus",
        "comma": "comma",
        "minus": "minus",
        "dot": "period",
        "slash": "slash",
        "zero": "zero",
        "one": "one",
        "two": "two",
        "three": "three",
        "four": "four",
        "five": "five",
        "six": "six",
        "seven": "seven",
        "eight": "eight",
        "nine": "nine",
        "colon": "colon",
        "semi": "semicolon",
        "lessthan": "less",
        "equal": "equal",
        "grthan": "greater",
        "ques_mrk": "question",
        "atsign": "at",
        "l_brack": "bracketleft",
        "backslsh": "backslash",
        "r_brack": "bracketright",
        "cirflex": "asciicircum",
        "l_quote": "quoteleft",
        "l_brace": "braceleft",
        "vert_bar": "bar",
        "r_brace": "braceright",
        "tilde": "asciitilde",
    }
    for i in range(26):
        name_map[chr(ord("A") + i)] = chr(ord("A") + i)
        name_map[chr(ord("a") + i)] = chr(ord("a") + i)

    with open(input_path, "r") as f:
        lines = f.readlines()

    glyph_chunks = get_glyph_chunks(lines)
    tmpdir = tempfile.mkdtemp()
    try:
        worker_func = partial(process_glyph_chunk, tmpdir=tmpdir, name_map=name_map)

        print(f"Processing {len(glyph_chunks)} glyphs...", flush=True)
        glyph_sfd_paths = []
        failed_glyphs = []
        with multiprocessing.Pool(1) as pool:
            results = [
                pool.apply_async(worker_func, (chunk,)) for chunk in glyph_chunks
            ]
            for res, chunk in zip(results, glyph_chunks):
                try:
                    result = res.get(timeout=1)
                    if result:
                        glyph_sfd_paths.append(result)
                    else:
                        glyph_name = chunk[0].strip()[:-1]
                        failed_glyphs.append(glyph_name)
                except multiprocessing.TimeoutError:
                    glyph_name = chunk[0].strip()[:-1]
                    print(
                        f"Glyph processing timeout for {font_name_base}: {glyph_name}",
                        flush=True,
                    )
                    failed_glyphs.append(glyph_name)

        font = fontforge.font()
        font.fontname = font_name
        font.familyname = family_name
        font.fullname = family_name
        font.weight = "Regular"
        font.copyright = f"Converted from Multics {font_name_base}"
        font.em = 1000

        successful_glyphs = 0
        for sfd_path in glyph_sfd_paths:
            if sfd_path and os.path.exists(sfd_path):
                try:
                    font.mergeFonts(sfd_path)
                    successful_glyphs += 1
                except Exception as e:
                    print(f"Could not merge font {sfd_path}: {e}", flush=True)

        if successful_glyphs > 0:
            font.save(output_path)
            print(
                f"Font saved to {output_path} with {successful_glyphs} glyphs.",
                flush=True,
            )
            if failed_glyphs:
                glyph_str = "glyph" if len(failed_glyphs) == 1 else "glyphs"
                error_str = "error" if len(failed_glyphs) == 1 else "errors"
                print(
                    f"Glyph processing {error_str} for {font_name_base}: {len(failed_glyphs)} {glyph_str} failed ({', '.join(failed_glyphs)})",
                    flush=True,
                )
        else:
            print("No glyphs were processed successfully.", flush=True)
    finally:
        shutil.rmtree(tmpdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert GCT font to SFD.")
    parser.add_argument("input", help="Input GCT file path.")
    parser.add_argument("output", help="Output SFD file path.")
    args = parser.parse_args()

    convert_gct_to_sfd(args.input, args.output)
