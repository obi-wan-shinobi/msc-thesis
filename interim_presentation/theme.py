from manim import *


def title_slide(scene, title, subtitle=None):
    t = Text(title, font_size=54).to_edge(UP)
    if subtitle:
        s = Text(subtitle, font_size=28).next_to(t, DOWN, buff=0.3)
        scene.play(FadeIn(t), FadeIn(s))
    else:
        scene.play(FadeIn(t))


def bullets_slide(scene, title, bullets):
    header = Text(title, font_size=44).to_edge(UP)
    items = (
        VGroup(*[Text(f"• {b}", font_size=30) for b in bullets])
        .arrange(DOWN, aligned_edge=LEFT, buff=0.28)
        .next_to(header, DOWN, buff=0.6)
        .to_edge(LEFT)
    )
    scene.play(FadeIn(header))
    for it in items:
        scene.play(FadeIn(it), run_time=0.25)


def figure_slide(scene, title, img_path, scale=1.0):
    header = Text(title, font_size=44).to_edge(UP)
    fig = ImageMobject(img_path).scale(scale).next_to(header, DOWN, buff=0.5)
    scene.play(FadeIn(header), FadeIn(fig))


def add_logo(scene, img_path="assets/TUDelft_logo_white.png", scale=0.15):
    logo = ImageMobject(img_path)
    logo.scale(scale)
    logo.to_corner(DOWN + LEFT, buff=0.2)
    scene.add(logo)
    return logo
