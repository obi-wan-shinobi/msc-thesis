from manim import *
from manim_slides import Slide

from theme import BG_COLOR, BODY_FS, HEADER_FS, MATH_FS, TEXT_COLOR, add_logo
from utils import *

try:
    from manim_slides import ThreeDSlide
except Exception:
    # Fallback: this will NOT have next_slide(), but at least makes the 3D code explicit.
    # Prefer fixing your imports to use manim_slides.ThreeDSlide.
    ThreeDSlide = ThreeDScene


class NTK(ThreeDSlide):
    def construct(self):
        # -----------------------------
        # Global styling
        # -----------------------------
        self.camera.background_color = BG_COLOR
        Text.set_default(color=TEXT_COLOR)
        MathTex.set_default(color=TEXT_COLOR)

        self.clear()

        # -----------------------------
        # 2D overlay UI (must NOT rotate with 3D camera)
        # -----------------------------
        logo = add_logo(self)

        title = Text(
            "Linearizing a function",
            font_size=HEADER_FS,
            weight=BOLD,
        ).to_edge(UP)

        # ------------------------------------------------------------
        # LEFT COLUMN: linearization / NTK dynamics (2D, fixed in frame)
        # ------------------------------------------------------------
        left_x_buff = 0.85

        t1 = Text(
            "The map",
            font_size=BODY_FS * 0.70,
            color=TEXT_COLOR,
        )
        m1 = MathTex(
            r"\theta \mapsto a\,\varphi(w^\top x),\qquad \theta=(a,w)",
            font_size=MATH_FS * 0.8,
            color=TEXT_COLOR,
        )

        t2 = Text(
            "So we linearize around a point:",
            font_size=BODY_FS * 0.70,
            color=TEXT_COLOR,
        )
        m2 = MathTex(
            r"f(\theta)\approx f(\theta_0)+\nabla_\theta f(\theta_0)\,(\theta-\theta_0)",
            font_size=MATH_FS * 0.8,
            color=TEXT_COLOR,
        )

        t3 = Text(
            "Then the residual dynamics become linear:",
            font_size=BODY_FS * 0.70,
            color=TEXT_COLOR,
        )
        m3 = MathTex(
            r"\dot r_t=-\Theta_t\,r_t",
            font_size=MATH_FS * 0.8,
            color=TEXT_COLOR,
        )
        m4 = MathTex(
            r"\textbf{NTK}:\,\Theta_t(x,y)=\nabla_\theta f(x;\theta_t)^\top\,\nabla_\theta f(y;\theta_t)",
            font_size=MATH_FS * 0.80,
            color=TEXT_COLOR,
        )

        # Use MathTex (not unicode) for the "width -> infinity" line
        m5 = MathTex(
            r"\text{As width } m\to\infty,\ \Theta_t\approx \Theta_0\ \text{ (approximately constant kernel).}",
            font_size=MATH_FS * 0.74,
            color=TEXT_COLOR,
        )

        left_block = VGroup(
            t1,
            m1,
            t2,
            m2,
            t3,
            m3,
            m4,
            m5,
        ).arrange(DOWN, buff=0.20, aligned_edge=LEFT)

        left_block.to_edge(LEFT, buff=left_x_buff)

        # Make sure this entire block does NOT rotate with the 3D camera
        self.add_fixed_in_frame_mobjects(left_block)

        # Start hidden; we’ll reveal line-by-line
        for mob in left_block:
            mob.set_opacity(0)

        # Keep logo/title screen-aligned
        self.add_fixed_in_frame_mobjects(logo, title)

        self.play(FadeIn(title, shift=UP * 0.2), run_time=0.6)
        self.next_slide()

        # -----------------------------
        # 3D camera orientation (rotates only 3D objects)
        # -----------------------------
        self.set_camera_orientation(
            phi=65 * DEGREES, theta=-55 * DEGREES, gamma=0 * DEGREES
        )

        # -----------------------------
        # 3D axes (subtle) + bounding box
        # -----------------------------
        axes = ThreeDAxes(
            x_range=[-3, 3, 1],
            y_range=[-2.5, 2.5, 1],
            z_range=[0, 3.5, 1],
            x_length=7.0,
            y_length=5.0,
            z_length=3.6,
            tips=False,
        )
        axes.set_stroke(color=TEXT_COLOR, width=1.0, opacity=0.30)

        # -----------------------------
        # Manifold surface (inverted bowl)
        # -----------------------------
        def f(x, y):
            base = 0.18 * (x**2 + 0.8 * y**2) + 0.10 * np.sin(0.9 * x) * np.cos(0.7 * y)
            return 2.6 - base  # inverted "cap"

        surf = Surface(
            lambda u, v: axes.c2p(u, v, f(u, v)),
            u_range=[-3, 3],
            v_range=[-2.5, 2.5],
            resolution=(28, 22),
            fill_opacity=0.35,
            stroke_width=0.6,
            stroke_opacity=0.22,
            color=BLUE_E,
        )

        # -----------------------------
        # Tangent plane at theta0
        # -----------------------------
        x0, y0 = -0.9, -0.4
        z0 = f(x0, y0)

        eps = 1e-3
        fx = (f(x0 + eps, y0) - f(x0 - eps, y0)) / (2 * eps)
        fy = (f(x0, y0 + eps) - f(x0, y0 - eps)) / (2 * eps)

        def plane_z(x, y):
            return z0 + fx * (x - x0) + fy * (y - y0)

        plane = Surface(
            lambda u, v: axes.c2p(u, v, plane_z(u, v)),
            u_range=[x0 - 2.4, x0 + 2.4],
            v_range=[y0 - 1.8, y0 + 1.8],
            resolution=(10, 8),
            fill_opacity=0.5,
            stroke_width=0.8,
            stroke_opacity=0.18,
            color=GRAY,  # different color than manifold; change as you like
        )

        # -----------------------------
        # Points and arrow (computed BEFORE shifting world)
        # -----------------------------
        try:
            from manim import Dot3D

            def dot3d(p, col=TEXT_COLOR, r=0.06):
                return Dot3D(p, radius=r, color=col)

        except Exception:

            def dot3d(p, col=TEXT_COLOR, r=0.06):
                return Sphere(radius=r, color=col).move_to(p)

        p0 = axes.c2p(x0, y0, z0)
        dot0 = dot3d(p0, col=TEXT_COLOR, r=0.06)

        x1, y1 = 1.3, 0.6
        p1 = axes.c2p(x1, y1, plane_z(x1, y1))
        dot1 = dot3d(p1, col=TEXT_COLOR, r=0.06)

        try:
            from manim import Arrow3D

            arrow = Arrow3D(p0, p1, color=RED, thickness=0.012)
        except Exception:
            arrow = Line3D(p0, p1, color=RED, thickness=0.012)

        # 3D-anchored labels that face the camera
        theta0_lbl = MathTex(r"\theta_0", font_size=BODY_FS, color=TEXT_COLOR).next_to(
            dot0, DOWN, buff=0.12
        )
        theta_lbl = MathTex(r"\theta", font_size=BODY_FS, color=TEXT_COLOR).next_to(
            dot1, RIGHT, buff=0.12
        )
        plane_lbl = Text(
            "tangent plane", font_size=BODY_FS * 0.75, color=TEXT_COLOR
        ).next_to(plane, UP, buff=0.15)

        # Keep labels facing the camera (but still positioned in 3D)
        self.add_fixed_orientation_mobjects(theta0_lbl, theta_lbl, plane_lbl)

        # -----------------------------
        # SHIFT THE WHOLE 3D WORLD DOWN (tweak this value)
        # -----------------------------
        world_scale = 0.5
        world_shift = DOWN * 2 + RIGHT * 4  # <-- adjust this

        world = VGroup(axes, surf, plane, dot0, dot1, arrow)
        world.scale(world_scale)
        world.shift(world_shift)

        # IMPORTANT: labels are also 3D-anchored -> shift them too
        theta0_lbl.shift(world_shift + RIGHT * 0.5)
        theta_lbl.shift(world_shift + LEFT * 0.7 + DOWN * 1.2)
        plane_lbl.shift(world_shift)

        # Animate the bullets
        t1.set_opacity(1)
        m1.set_opacity(1)
        self.play(FadeIn(t1), FadeIn(m1), run_time=0.45)
        self.next_slide()

        self.play(FadeIn(axes), run_time=0.6)
        self.next_slide()

        self.play(FadeIn(surf, shift=UP * 0.05), run_time=0.8)
        self.next_slide()

        t2.set_opacity(1)
        m2.set_opacity(1)
        self.play(FadeIn(t2), FadeIn(m2), run_time=0.75)
        self.next_slide()

        self.play(FadeIn(plane, shift=UP * 0.05), run_time=0.7)
        self.next_slide()

        self.play(FadeIn(dot0), FadeIn(theta0_lbl), run_time=0.4)
        self.next_slide()

        self.play(FadeIn(dot1), FadeIn(theta_lbl), run_time=0.4)
        self.next_slide()

        self.play(GrowFromPoint(arrow, dot0.get_center()), run_time=0.6)
        self.next_slide()

        self.play(FadeIn(plane_lbl, shift=UP * 0.05), run_time=0.35)
        self.next_slide()

        self.play(Indicate(plane, scale_factor=1.01), run_time=0.6)
        self.next_slide()

        t3.set_opacity(1)
        m3.set_opacity(1)
        m4.set_opacity(1)
        self.play(FadeIn(t3), FadeIn(m3), FadeIn(m4), run_time=0.75)
        self.next_slide()

        m5.set_opacity(1)
        self.play(FadeIn(m5), run_time=0.45)
