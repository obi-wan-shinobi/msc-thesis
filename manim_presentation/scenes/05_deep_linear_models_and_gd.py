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


class DeepLinearModelsAndGD(ThreeDSlide):
    """
    Deep linear network: f_{v,W}(x)=v^T(Wx).
    Same slide grammar as LinearModelsAndGD, but shows factorized params + nonconvex GD.
    """

    def construct(self):
        self.camera.background_color = BG_COLOR
        Text.set_default(color=TEXT_COLOR)
        MathTex.set_default(color=TEXT_COLOR)

        self.clear()
        logo = add_logo(self)

        label_color = BLUE

        # ---------------------------------------------------------------------
        # Title
        # ---------------------------------------------------------------------
        title = Text(
            "Deep linear models and gradient descent",
            font_size=HEADER_FS,
            weight=BOLD,
        ).to_edge(UP)

        self.play(FadeIn(title, shift=UP * 0.2), run_time=0.6)
        self.next_slide()

        # ---------------------------------------------------------------------
        # LEFT COLUMN — Setup
        # ---------------------------------------------------------------------
        setup_lab = Text("Setup:", font_size=BODY_FS, color=label_color, weight=BOLD)
        setup_eq = MathTex(
            r"f_{v,W}(x)=v^\top(Wx),\quad W\in\mathbb{R}^{m\times d},\ v\in\mathbb{R}^m",
            font_size=MATH_FS,
        )
        setup = VGroup(setup_lab, setup_eq).arrange(RIGHT, buff=0.45, aligned_edge=UP)
        setup.next_to(title, DOWN, buff=0.6).to_edge(LEFT, buff=0.9)

        self.play(Write(setup_lab), run_time=0.35)
        self.play(Write(setup_eq), run_time=0.8)
        self.next_slide()

        # ---------------------------------------------------------------------
        # RIGHT COLUMN — Network (top-right) as anchor (deep linear only)
        # ---------------------------------------------------------------------
        neuron_stroke = GRAY_B
        edge_color = GRAY_C
        m = 10

        nn = NetworkMobject(
            layer_sizes=(7, m, 1),
            neuron_radius=0.095,
            neuron_to_neuron_buff=0.28,
            layer_to_layer_buff=1.35,
            neuron_stroke_color=neuron_stroke,
            neuron_stroke_width=2.2,
            neuron_fill_color=BLUE_E,
            neuron_fill_opacity=0.0,
            edge_color=edge_color,
            edge_stroke_width=1.15,
            edge_propagation_color=YELLOW,
            edge_propagation_time=0.55,
            brace_for_large_layers=False,
        ).deactivate()

        nn.scale(0.72)
        nn.to_edge(RIGHT, buff=0.85).shift(UP * 1.25)
        nn.shift(LEFT * 1.25)  # leave room for inset

        x_lbl = MathTex(r"x", font_size=BODY_FS).next_to(nn.layers[0], LEFT, buff=0.18)
        fx_lbl = MathTex(r"f_{v,W}(x)", font_size=BODY_FS).next_to(
            nn.layers[-1], RIGHT, buff=0.18
        )

        # place W under first edge bundle, v under second edge bundle
        W_lbl = MathTex(r"W", font_size=BODY_FS, color=label_color).move_to(
            nn.get_center() + DOWN * 1.5 + LEFT * 0.55
        )
        v_lbl = MathTex(r"v", font_size=BODY_FS, color=label_color).move_to(
            nn.get_center() + DOWN * 1 + RIGHT * 0.55
        )

        self.play(
            FadeIn(nn, shift=RIGHT * 0.15),
            FadeIn(x_lbl, shift=RIGHT * 0.1),
            FadeIn(fx_lbl, shift=LEFT * 0.1),
            FadeIn(W_lbl, shift=UP * 0.1),
            FadeIn(v_lbl, shift=UP * 0.1),
            run_time=0.6,
        )
        self.next_slide()

        # quick deterministic pulse
        acts = [
            np.full(len(nn.layers[0].neurons), 0.30),
            np.full(len(nn.layers[1].neurons), 0.55),
            np.array([0.75]),
        ]
        self.play(nn.layer_activate_anim(0, acts[0], run_time=0.20), run_time=0.20)
        self.play(nn.forward_pass_anim(activations=acts), run_time=0.9)
        self.next_slide()
        self.play(
            nn.layer_activate_anim(2, np.array([0.0]), run_time=0.22), run_time=0.25
        )
        self.next_slide()

        # ---------------------------------------------------------------------
        # LEFT COLUMN — Loss
        # ---------------------------------------------------------------------
        loss_lab = Text("Loss:", font_size=BODY_FS, color=label_color, weight=BOLD)
        loss_eq = MathTex(
            r"\mathcal{L}(v,W)=\tfrac12\|X^\top W^\top v-y\|_2^2",
            font_size=MATH_FS,
        )
        loss = VGroup(loss_lab, loss_eq).arrange(RIGHT, buff=0.45, aligned_edge=UP)
        loss.next_to(setup, DOWN, buff=0.35).align_to(setup, LEFT)

        self.play(Write(loss_lab), run_time=0.35)
        self.play(Write(loss_eq), run_time=0.75)
        self.next_slide()

        # ---------------------------------------------------------------------
        # LEFT COLUMN — GD update as a HEADER line, equations start BELOW it
        # ---------------------------------------------------------------------
        upd_lab = Text("GD update:", font_size=BODY_FS, color=label_color, weight=BOLD)
        upd_lab.next_to(loss, DOWN, buff=0.42).align_to(setup, LEFT)
        self.play(Write(upd_lab), run_time=0.35)
        self.next_slide()

        v_upd = MathTex(
            r"v_{t+1}=v_t-\eta\,\nabla_v\mathcal{L}(v_t,W_t)",
            font_size=MATH_FS * 0.82,
        )
        W_upd = MathTex(
            r"W_{t+1}=W_t-\eta\,\nabla_W\mathcal{L}(v_t,W_t)",
            font_size=MATH_FS * 0.82,
        )

        upd_generic = VGroup(v_upd, W_upd).arrange(DOWN, buff=0.45, aligned_edge=UP)
        upd_generic.next_to(upd_lab, DOWN, buff=0.18).align_to(loss_eq, LEFT)
        self.play(Write(upd_generic), run_time=0.85)
        self.next_slide()

        grad_v = MathTex(
            r"\nabla_v\mathcal{L}(v_t,W_t)=W_tX\,(X^\top W_t^\top v_t - y)",
            font_size=MATH_FS * 0.84,
        )
        grad_W = MathTex(
            r"\nabla_W\mathcal{L}(v_t,W_t)=v_t\,(X^\top W_t^\top v_t - y)^\top X^\top",
            font_size=MATH_FS * 0.84,
        )
        grad_v.next_to(upd_generic, DOWN, buff=0.45).align_to(upd_generic, LEFT)
        grad_W.next_to(grad_v, DOWN, buff=0.45).align_to(upd_generic, LEFT)

        self.play(FadeIn(grad_v, shift=UP * 0.05), run_time=0.45)
        self.play(FadeIn(grad_W, shift=UP * 0.05), run_time=0.45)
        self.next_slide()

        # coupling highlight (visual feedback loop)
        # boxW = SurroundingRectangle(
        #     grad_v.get_part_by_tex("W_t"), buff=0.06, color=YELLOW
        # )
        # boxv = SurroundingRectangle(
        #     grad_W.get_part_by_tex("v_t"), buff=0.06, color=YELLOv
        # )
        # self.play(Create(boxW), run_time=0.25)
        # self.play(Transform(boxW, boxv), run_time=0.35)
        # self.play(FadeOut(boxW), run_time=0.2)
        # self.next_slide()

        # ---------------------------------------------------------------------
        # Freeze all 2D content before tilting camera for 3D inset
        # ---------------------------------------------------------------------
        fixed_2d = Group(
            title,
            logo,
            setup_lab,
            setup_eq,
            nn,
            x_lbl,
            fx_lbl,
            W_lbl,
            v_lbl,
            loss_lab,
            loss_eq,
            upd_lab,
            upd_generic,
            grad_v,
            grad_W,
        )
        self.add_fixed_in_frame_mobjects(fixed_2d)

        # Gentle 3D angle (readable; no drama)
        self.set_camera_orientation(phi=65 * DEGREES, theta=-45 * DEGREES)

        # ---------------------------------------------------------------------
        # Bottom-right inset — 3D loss surface
        # $$L(u,v) = A_1\sin(f_1u) + A_2\cos(f_2v) + A_3\sin(f_3(u + v))$$
        # ---------------------------------------------------------------------
        A_1, A_2, A_3 = 1.5, 1.2, 3.0
        f_1, f_2, f_3 = 0.4, 0.3, 0.05

        def L_uv(u, v):
            return (
                A_1 * np.sin(f_1 * u)
                + A_2 * np.cos(f_2 * v)
                + A_3 * np.sin(f_3 * (u + v))
            )

        # same slot geometry as your previous inset
        slot = Rectangle(width=3.7, height=2.2, stroke_opacity=0.0)
        slot.to_corner(DR, buff=0.85).shift(UP * 0.25)

        # domain
        umin, umax = -10.0, 10.0
        vmin, vmax = -10.0, 10.0

        # pick z-range that actually matches your amplitudes (~[-4.2, 4.2])
        zmin, zmax = -4.5, 4.5

        axes = ThreeDAxes(
            x_range=[umin, umax, 5],
            y_range=[vmin, vmax, 5],
            z_range=[zmin, zmax, 1.5],
            x_length=3.0,  # compact
            y_length=2.6,
            z_length=1.6,
            tips=False,
        )
        # Axis labels (part of the inset, not fixed in frame)
        x_lab3d = MathTex("u", font_size=BODY_FS * 0.9).next_to(
            axes.x_axis.get_end(), DOWN, buff=0.08
        )
        y_lab3d = MathTex("\mathcal{L}(u,v)", font_size=BODY_FS * 0.9).next_to(
            axes.y_axis.get_end(), LEFT, buff=0.08
        )
        z_lab3d = (
            MathTex("v", font_size=BODY_FS * 0.9)
            .next_to(axes.z_axis.get_end(), UP, buff=0.08)
            .shift(UP * 1 + RIGHT * 0.5)
        )

        axis_labels = VGroup(x_lab3d, y_lab3d, z_lab3d).set_color(GRAY_A)
        # make axes subtle (optional)
        axes.set_stroke(width=1.2, opacity=0.7)

        # IMPORTANT: use axes.c2p so surface is in the same coordinate system as the axes
        surface = Surface(
            lambda u, v: axes.c2p(u, v, L_uv(u, v)),
            u_range=[umin, umax],
            v_range=[vmin, vmax],
            resolution=(80, 80),
            fill_opacity=0.60,
            stroke_width=0.0,
            stroke_opacity=0.0,
            color=BLUE_E,
        )

        # (optional) if you want axis labels inside the inset, keep them; otherwise remove
        # axis_labels = axes.get_axis_labels(x_label="u", y_label="v", z_label="L")
        # inset3d = VGroup(axes, surface, axis_labels)

        inset3d = VGroup(axes, surface, axis_labels)

        # Fit inset group into slot (NOT just surface)
        s = min(slot.width / inset3d.width, slot.height / inset3d.height)
        inset3d.scale(s)

        # Place into slot and nudge right
        inset3d.move_to(slot.get_center()).shift(RIGHT * 1.25 + UP * 1.5)
        self.add_fixed_orientation_mobjects(axis_labels)

        # Now add/animate the inset as one object
        self.play(FadeIn(inset3d, shift=UP * 1.2), run_time=0.6)
        self.next_slide()
