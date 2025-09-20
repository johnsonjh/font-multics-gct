#!/usr/bin/env python
import fontforge
import os
import re

# --- Constants ---
SCALE = 20
Y_OFFSET = 0
STROKE_WIDTH = 15

def convert_gct_to_sfd(input_path, output_path):
    font = fontforge.font()
    font.fontname = "GCTGothicEnglish"
    font.familyname = "GCT Gothic English"
    font.fullname = "GCT Gothic English"
    font.weight = "Regular"
    font.copyright = "Converted from Multics gct_gothic_english"
    font.em = 1000

    name_map = {
        "bel": "bell", "excl_pt": "exclam", "dbl_quot": "quotedbl",
        "sharp": "numbersign", "dollar": "dollar", "percent": "percent",
        "amprsnd": "ampersand", "r_quote": "quoteright", "l_paren": "parenleft",
        "r_paren": "parenright", "star": "asterisk", "plus": "plus",
        "comma": "comma", "minus": "minus", "dot": "period", "slash": "slash",
        "zero": "zero", "one": "one", "two": "two", "three": "three", "four": "four",
        "five": "five", "six": "six", "seven": "seven", "eight": "eight", "nine": "nine",
        "colon": "colon", "semi": "semicolon", "lessthan": "less",
        "equal": "equal", "grthan": "greater", "ques_mrk": "question",
        "atsign": "at", "l_brack": "bracketleft", "backslsh": "backslash",
        "r_brack": "bracketright", "cirflex": "asciicircum", "l_quote": "quoteleft",
        "l_brace": "braceleft", "vert_bar": "bar", "r_brace": "braceright",
        "tilde": "asciitilde"
    }
    for i in range(26):
        name_map[chr(ord('A') + i)] = chr(ord('A') + i)
        name_map[chr(ord('a') + i)] = chr(ord('a') + i)

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
                else: # command == "end"
                    if current_glyph:
                        # Draw all collected strokes as open contours
                        for stroke_points in all_strokes:
                            if not stroke_points or len(stroke_points) < 2:
                                continue
                            scaled_points = [(p[0] * SCALE, p[1] * SCALE + Y_OFFSET) for p in stroke_points]
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
                        # Move the finished glyph into its final position
                        current_glyph.transform((1, 0, 0, 1, 0, 355))

                    in_glyph = False
                    current_glyph = None
            
            elif command == "vector":
                if not current_stroke:
                    current_stroke.append((x,y))
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
    input_file = os.path.join(os.path.dirname(__file__), "gct_gothic_english_")
    output_file = os.path.join(os.path.dirname(__file__), "GCTGothicEnglish.sfd")
    convert_gct_to_sfd(input_file, output_file)
