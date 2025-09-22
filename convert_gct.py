#!/usr/bin/env python3
import fontforge
import os
import re
import argparse

# --- Constants ---
SCALE = 20
Y_OFFSET = 200
STROKE_WIDTH = 15


def gct_name_to_font_name(gct_name):
    # gct_gothic_english_ -> GCTGothicEnglish
    base_name = gct_name.replace("gct_", "").strip("_")
    parts = base_name.split("_")
    return "GCT" + "".join([p.capitalize() for p in parts])


def gct_name_to_family_name(gct_name):
    # gct_gothic_english_ -> GCT Gothic English
    base_name = gct_name.replace("gct_", "").strip("_")
    parts = base_name.split("_")
    return "GCT " + " ".join([p.capitalize() for p in parts])


def convert_gct_to_sfd(input_path, output_path):
    font_name_base = os.path.basename(input_path)
    font_name = gct_name_to_font_name(font_name_base)
    family_name = gct_name_to_family_name(font_name_base)

    font = fontforge.font()
    font.fontname = font_name
    font.familyname = family_name
    font.fullname = family_name
    font.weight = "Regular"
    font.copyright = f"Converted from Multics {font_name_base}"
    font.em = 1000

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

    current_glyph = None
    x, y = 0, 0
    in_glyph = False
    all_strokes = []
    current_stroke = []

    for line in lines:
        line = line.strip()
        if not line or line.startswith("metric"):
            continue

        if line.endswith(":"):
            glyph_name = line[:-1]
            std_name = name_map.get(glyph_name, glyph_name)

            current_glyph = font.createChar(-1, std_name)
            print(f"Processing glyph: {glyph_name} -> {std_name}")

            x, y = 0, 0
            in_glyph = True
            all_strokes = []
            current_stroke = []
            if current_glyph:
                current_glyph.clear()

        elif in_glyph:
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
                        print(f"Warning: Could not parse coordinates in shift: {line}")
                else:  # command == "end"
                    if current_glyph:
                        # Draw all collected strokes as open contours
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

                        # Finalize the glyph
                        current_glyph.width = x * SCALE
                        current_glyph.stroke("circular", STROKE_WIDTH)
                        current_glyph.removeOverlap()
                        current_glyph.correctDirection()

                    in_glyph = False
                    current_glyph = None

            elif command == "vector":
                if not current_stroke:
                    current_stroke.append((x, y))
                try:
                    dx, dy = int(parts[1]), int(parts[2])
                    x += dx
                    y += dy
                    current_stroke.append((x, y))
                except (ValueError, IndexError):
                    print(f"Warning: Could not parse coordinates in vector: {line}")

    font.save(output_path)
    print(f"Font saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert GCT font to SFD.")
    parser.add_argument("input", help="Input GCT file path.")
    parser.add_argument("output", help="Output SFD file path.")
    args = parser.parse_args()

    convert_gct_to_sfd(args.input, args.output)