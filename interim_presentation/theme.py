from manim import *

BG_COLOR = WHITE
TEXT_COLOR = BLACK if BG_COLOR == WHITE else WHITE
LOGO_PATH = (
    "assets/TUDelft_logo_black.png"
    if BG_COLOR == WHITE
    else "assets/TUDelft_logo_white.png"
)

TITLE_FS = 56
HEADER_FS = 44
SUBTITLE_FS = 28
BODY_FS = 32
MATH_FS = 36


def add_logo(scene, img_path=LOGO_PATH, scale=0.15):
    logo = ImageMobject(img_path)
    logo.scale(scale)
    logo.to_corner(DOWN + LEFT, buff=0.2)
    scene.add(logo)
    return logo
