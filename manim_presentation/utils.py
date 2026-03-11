import itertools as it

import numpy as np
from manim import *


class NetworkMobject(VGroup):
    def __init__(
        self,
        layer_sizes=(3, 7, 5, 2),
        neuron_radius=0.11,
        neuron_to_neuron_buff=0.34,
        layer_to_layer_buff=1.25,
        neuron_stroke_color=GRAY_B,
        neuron_stroke_width=2,
        neuron_fill_color=BLUE_E,
        neuron_fill_opacity=0.0,
        edge_color=GRAY_C,
        edge_stroke_width=1.2,
        edge_propagation_color=YELLOW,
        edge_propagation_time=0.7,
        max_shown_neurons=16,
        brace_for_large_layers=False,  # keep False (no braces)
        average_shown_activation_of_large_layer=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.layer_sizes = list(layer_sizes)

        self.neuron_radius = neuron_radius
        self.neuron_to_neuron_buff = neuron_to_neuron_buff
        self.layer_to_layer_buff = layer_to_layer_buff

        self.neuron_stroke_color = neuron_stroke_color
        self.neuron_stroke_width = neuron_stroke_width
        self.neuron_fill_color = neuron_fill_color
        self.neuron_fill_opacity = neuron_fill_opacity

        self.edge_color = edge_color
        self.edge_stroke_width = edge_stroke_width
        self.edge_propagation_color = edge_propagation_color
        self.edge_propagation_time = edge_propagation_time

        self.max_shown_neurons = max_shown_neurons
        self.brace_for_large_layers = brace_for_large_layers
        self.average_shown_activation_of_large_layer = (
            average_shown_activation_of_large_layer
        )

        self._build()

    # ------------------------
    # Build
    # ------------------------
    def _build(self):
        self.layers = VGroup(*[self._get_layer(sz) for sz in self.layer_sizes])
        self.layers.arrange(RIGHT, buff=self.layer_to_layer_buff)

        self.edge_groups = VGroup()
        for l1, l2 in zip(self.layers[:-1], self.layers[1:]):
            edge_group = VGroup()
            for n1, n2 in it.product(l1.neurons, l2.neurons):
                e = self._get_edge(n1, n2)
                edge_group.add(e)
                n1.edges_out.add(e)
                n2.edges_in.add(e)
            self.edge_groups.add(edge_group)

        # Ensure edges behind neurons
        self.edge_groups.set_z_index(0)
        self.layers.set_z_index(1)

        self.add(self.edge_groups, self.layers)

    def _get_layer(self, size: int):
        layer = VGroup()

        n_show = min(size, self.max_shown_neurons)

        neurons = VGroup(
            *[
                Circle(
                    radius=self.neuron_radius,
                    stroke_color=self.neuron_stroke_color,
                    stroke_width=self.neuron_stroke_width,
                    fill_color=self.neuron_fill_color,
                    fill_opacity=self.neuron_fill_opacity,
                )
                for _ in range(n_show)
            ]
        ).arrange(DOWN, buff=self.neuron_to_neuron_buff)

        for neuron in neurons:
            neuron.edges_in = VGroup()
            neuron.edges_out = VGroup()

        layer.neurons = neurons
        layer.add(neurons)

        # If you ever use very large layers, you can optionally add dots/brace,
        # but for your sizes (3,7,5,2) this never triggers.
        if size > n_show:
            dots = Tex(r"\vdots").set_color(self.neuron_stroke_color)
            dots.move_to(neurons)
            layer.add(dots)
            layer.dots = dots

            if self.brace_for_large_layers:
                brace = Brace(layer, LEFT)
                brace_label = brace.get_tex(str(size))
                layer.add(brace, brace_label)
                layer.brace = brace
                layer.brace_label = brace_label

        return layer

    def _get_edge(self, neuron1, neuron2):
        return Line(
            neuron1.get_center(),
            neuron2.get_center(),
            buff=self.neuron_radius,
            stroke_color=self.edge_color,
            stroke_width=self.edge_stroke_width,
        )

    # ------------------------
    # Activation helpers
    # ------------------------
    def deactivate(self):
        for layer in self.layers:
            for n in layer.neurons:
                n.set_fill(opacity=0.0)
        return self

    def _compress_activation(self, av, n_show):
        """If activation vector longer than shown neurons, compress it."""
        av = np.array(av, dtype=float).ravel()
        if len(av) <= n_show:
            return av

        if not self.average_shown_activation_of_large_layer:
            half = n_show // 2
            return np.concatenate([av[:half], av[-(n_show - half) :]])

        # bucket-average into n_show bins (simple, stable)
        idx = (np.linspace(0, len(av), n_show + 1)).astype(int)
        out = []
        for a, b in zip(idx[:-1], idx[1:]):
            chunk = av[a:b] if b > a else av[a : a + 1]
            out.append(float(np.mean(chunk)))
        return np.array(out)

    def layer_activate_anim(self, layer_index, activation_vector=None, run_time=0.4):
        """
        Returns an Animation that sets neuron fill opacities.
        activation_vector in [0,1]. If None, random-ish.
        """
        layer = self.layers[layer_index]
        n_show = len(layer.neurons)

        if activation_vector is None:
            activation_vector = np.random.uniform(0.15, 0.85, size=n_show)
        else:
            activation_vector = self._compress_activation(activation_vector, n_show)
            if len(activation_vector) < n_show:
                activation_vector = np.pad(
                    activation_vector, (0, n_show - len(activation_vector)), mode="edge"
                )

        # Build per-neuron animations (so you see variation)
        anims = []
        for a, neuron in zip(activation_vector, layer.neurons):
            anims.append(neuron.animate.set_fill(opacity=float(np.clip(a, 0, 1))))
        return AnimationGroup(*anims, lag_ratio=0.05, run_time=run_time)

    def edge_propagation_anim(self, edge_group_index, run_time=None):
        eg = self.edge_groups[edge_group_index].copy()
        eg.set_stroke(self.edge_propagation_color, width=1.6 * self.edge_stroke_width)

        return ShowPassingFlash(
            eg,
            time_width=0.35,  # tweak: thickness of the “moving flash”
            run_time=run_time or self.edge_propagation_time,
        )

    def forward_pass_anim(self, activations=None):
        """
        activations: optional list (per-layer) of activation vectors.
        If omitted, uses random activations for hidden/output layers.
        """
        n_layers = len(self.layers)
        acts = activations or [None] * n_layers

        # Typically leave input layer off / faint:
        self.layers[0].set_opacity(1.0)

        steps = []
        for i in range(n_layers - 1):
            steps.append(
                AnimationGroup(
                    self.edge_propagation_anim(i),
                    self.layer_activate_anim(i + 1, acts[i + 1], run_time=0.35),
                    lag_ratio=0.15,
                )
            )
        return Succession(*steps)
