from manim import *
from manim_slides import Slide

from theme import BG_COLOR, BODY_FS, HEADER_FS, MATH_FS, TEXT_COLOR, add_logo
from utils import *


class LemmaConcentrationBounds(Slide):
    def construct(self):
        # -----------------------------
        # Global styling
        # -----------------------------
        self.camera.background_color = BG_COLOR
        Text.set_default(color=TEXT_COLOR)
        MathTex.set_default(color=TEXT_COLOR)

        self.clear()
        logo = add_logo(self)  # don't move

        label_color = BLUE
        math_fs = MATH_FS * 0.78
        row_lab_fs = BODY_FS * 0.75
        txt_fs = BODY_FS * 0.72

        # -----------------------------
        # Title
        # -----------------------------
        title = Text(
            "Finite samples: concentration",
            font_size=HEADER_FS * 0.92,
            weight=BOLD,
        ).to_edge(UP)

        self.add(logo)
        self.play(FadeIn(title, shift=UP * 0.2), run_time=0.6)
        self.next_slide()

        # -----------------------------
        # Helper: one-line row (label + body)
        # -----------------------------
        def row(label: str, body: Mobject, buff=0.55):
            lab = Text(label, font_size=row_lab_fs, color=label_color, weight=BOLD)
            return VGroup(lab, body).arrange(RIGHT, buff=buff, aligned_edge=UP)

        # -----------------------------
        # Lead + notation (left aligned)
        # -----------------------------
        notation = row(
            "Notation:",
            MathTex(
                r"d=2K+1,\qquad \kappa=\sup_{\delta\in[0,2\pi)}|\Theta(\delta)|",
                font_size=math_fs * 0.95,
            ),
            buff=0.50,
        )
        notation.next_to(title, DOWN, buff=0.30).to_edge(LEFT, buff=0.85)

        self.play(FadeIn(notation, shift=UP * 0.05), run_time=0.45)
        self.next_slide()

        # ============================================================
        # Lemma A: bound -> takeaway
        # ============================================================
        A_bound = VGroup(
            MathTex(r"\mathbb{E}[G]=I_{d}", font_size=math_fs),
            MathTex(
                r"\mathbb{P}\!\left(\|G-I_{d}\|_{\max}\ge \varepsilon\right)"
                r"\le 2d^{2}\exp\!\left(-\frac{n\varepsilon^2}{8}\right)",
                font_size=math_fs * 0.95,
            ),
        ).arrange(DOWN, buff=0.16, aligned_edge=LEFT)

        A_row_bound = row("Lemma A:", A_bound)
        A_row_bound.next_to(notation, DOWN, buff=0.38).to_edge(LEFT, buff=0.85)

        self.play(FadeIn(A_row_bound, shift=UP * 0.05), run_time=0.6)
        self.next_slide()

        A_take = Text(
            "Sampled Fourier features are almost orthonormal (G ≈ I).",
            font_size=txt_fs,
            color=TEXT_COLOR,
        )
        A_row_take = row("Lemma A:", A_take)
        A_row_take.move_to(A_row_bound).align_to(A_row_bound, LEFT)

        self.play(FadeTransform(A_row_bound, A_row_take), run_time=0.6)
        self.next_slide()

        # ============================================================
        # Lemma B: bound -> takeaway
        # ============================================================
        B_bound = VGroup(
            MathTex(
                r"H=\frac{1}{n}\Phi^\top A\,\Phi",
                font_size=math_fs,
            ),
            MathTex(
                r"\lambda_{p}^{(n)}=\Bigl(1-\frac{1}{n}\Bigr)\lambda_p+\frac{1}{n}\Theta(0),"
                r"\qquad \mathbb{E}[H]=\Lambda^{(n)}",
                font_size=math_fs * 0.90,
            ),
            MathTex(
                r"\mathbb{P}\!\left(\|H-\mathbb{E}H\|_{\max}\ge \varepsilon\right)"
                r"\le 2d^{2}\exp\!\left(-\frac{n\varepsilon^2}{32\kappa^2}\right)",
                font_size=math_fs * 0.90,
            ),
        ).arrange(DOWN, buff=0.14, aligned_edge=LEFT)

        B_row_bound = row("Lemma B:", B_bound)
        B_row_bound.next_to(A_row_take, DOWN, buff=0.42).align_to(A_row_take, LEFT)

        self.play(FadeIn(B_row_bound, shift=UP * 0.05), run_time=0.6)
        self.next_slide()

        B_take = Text(
            "The compressed operator is almost diagonal on low modes (H ≈ Λ).",
            font_size=txt_fs,
            color=TEXT_COLOR,
        )
        B_row_take = row("Lemma B:", B_take)
        B_row_take.move_to(B_row_bound).align_to(B_row_bound, LEFT)

        self.play(FadeTransform(B_row_bound, B_row_take), run_time=0.6)
        self.next_slide()

        # ============================================================
        # Lemma C: bound -> takeaway
        # ============================================================
        C_bound = MathTex(
            r"\mathbb{P}\!\left(\|A\Phi-\Phi\Lambda^{(n)}\|_{\max}\ge \varepsilon\right)"
            r"\le 2nd\,\exp\!\left(-\frac{n\varepsilon^2}{4\kappa^2}\right),\qquad (n\ge 2)",
            font_size=math_fs * 0.90,
        )
        C_row_bound = row("Lemma C:", C_bound)
        C_row_bound.next_to(B_row_take, DOWN, buff=0.42).align_to(B_row_take, LEFT)

        self.play(FadeIn(C_row_bound, shift=UP * 0.05), run_time=0.6)
        self.next_slide()

        C_take = Text(
            "Low Fourier modes act like eigenvectors (AΦ ≈ ΦΛ), uniformly).",
            font_size=txt_fs,
            color=TEXT_COLOR,
        )
        C_row_take = row("Lemma C:", C_take)
        C_row_take.move_to(C_row_bound).align_to(C_row_bound, LEFT)

        self.play(FadeTransform(C_row_bound, C_row_take), run_time=0.6)
        self.next_slide()

        # ============================================================
        # Final key line (keep above logo region)
        # ============================================================
        key = Text(
            "Low-mode diagonalization and decay rates survive finite sampling.",
            font_size=BODY_FS * 0.70,
            color=TEXT_COLOR,
        )
        key.next_to(C_row_take, DOWN, buff=0.45).align_to(C_row_take, LEFT)
        key.shift(RIGHT * 0.6)  # avoid TU Delft logo area

        self.play(FadeIn(key, shift=UP * 0.05), run_time=0.5)
        self.next_slide()
