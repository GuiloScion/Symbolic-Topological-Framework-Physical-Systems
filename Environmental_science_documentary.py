# main.py
"""
Advanced Manim documentary with on-screen subtitles for the full 30-minute script.
Designed for Manim Community v0.16+ / v0.17+ and Google Colab.

Drop 'narration.mp3' in workspace if you want to add audio later; subtitles will display regardless.
"""

from manim import *
import numpy as np
import random
from math import pi

# ---------------- CONFIG ----------------
config.pixel_width = 1920
config.pixel_height = 1080
config.frame_rate = 30
config.background_color = "#071018"  # dark background for cinematic look

# Total target duration (seconds) = 1800 (30 minutes)
SCENE_DURATIONS = {
    "IntroScene": 120,
    "HistoryScene": 180,
    "NetworkOverviewScene": 240,
    "MiningExchangeScene": 220,
    "CommunicationScene": 160,
    "ArchitectureScene": 120,
    "CompetitionScene": 100,
    "EcosystemsScene": 100,
    "ResearchMethodsScene": 150,
    "ThreatsRestorationScene": 200,
    "ConclusionScene": 210,
}

# ---------------- HELPERS ----------------
def subtitle_mobject(text: str, font_size=28, max_width=13):
    """Centered subtitle block with semi-opaque background"""
    p = Paragraph(*text.split("\n"), alignment="CENTER", font_size=font_size, weight=BOLD)
    p.set_color(WHITE)
    bg = SurroundingRectangle(p, buff=0.22, corner_radius=0.12)
    bg.set_fill(BLACK, opacity=0.65)
    bg.set_stroke(width=0)
    group = VGroup(bg, p).to_edge(DOWN, buff=0.7)
    group.set_z_index(2000)
    return group

def display_subtitles(scene: Scene, lines: list[str], total_time: float):
    """Display subtitle lines evenly over total_time in the provided scene."""
    if not lines:
        return
    per = total_time / len(lines)
    for i, txt in enumerate(lines):
        sub = subtitle_mobject(txt)
        # short fade in and fade out around main wait to keep flow smooth
        scene.play(FadeIn(sub, shift=UP * 0.2), run_time=0.6)
        # wait most of the allotted time
        # keep small fade durations so total doesn't drift much
        scene.wait(max(0, per - 0.9))
        scene.play(FadeOut(sub, shift=DOWN * 0.2), run_time=0.6)

# Simple drawing primitives used across scenes
def make_tree(size=1.0, trunk_color="#6b4f25", foliage_color="#1b6a31"):
    trunk = Rectangle(width=0.12 * size, height=0.7 * size, fill_color=trunk_color, fill_opacity=1)
    trunk.move_to(DOWN * 0.35 * size)
    leaves = VGroup()
    radii = [0.9 * size, 0.65 * size, 0.45 * size]
    offsets = [0.05 * size, 0.2 * size, 0.38 * size]
    for r, off in zip(radii, offsets):
        circ = Circle(radius=r, fill_color=foliage_color, fill_opacity=1, stroke_width=0)
        circ.shift(UP * off * size)
        leaves.add(circ)
    return VGroup(trunk, leaves)

class Hyphae(VMobject):
    """Fractal-like hyphae visual."""
    def __init__(self, start=ORIGIN, depth=5, length=1.0, angle_spread=pi/2.2, color="#19a7a4", stroke_width=2.0, **kwargs):
        super().__init__(**kwargs)
        self.start = np.array(start)
        self.depth = depth
        self.length = length
        self.angle_spread = angle_spread
        self.color = color
        self.stroke_width = stroke_width
        self._build()

    def _build(self):
        self.clear_points()
        segments = [(self.start, -pi/2 + random.uniform(-0.12, 0.12))]
        for d in range(self.depth):
            new_segments = []
            for p, ang in segments:
                n_children = 2 if d == 0 else random.choice([1, 2])
                for i in range(n_children):
                    a = ang + random.uniform(-self.angle_spread/2, self.angle_spread/2) * (1.0 / (d + 1))
                    seg_len = self.length * (0.9 ** d) * random.uniform(0.78, 1.05)
                    q = p + seg_len * np.array([np.cos(a), np.sin(a), 0])
                    line = Line(p, q)
                    line.set_stroke(self.color, width=self.stroke_width * (1.0 / (d + 0.9)))
                    self.append_vectorized_mobject(line)
                    new_segments.append((q, a))
            segments = new_segments

class NutrientParticles(VGroup):
    def __init__(self, path: VMobject, count=10, color=YELLOW, radius=0.04):
        super().__init__()
        self.path = path
        self.count = count
        for i in range(count):
            dot = Dot(radius=radius, color=color)
            alpha = i / count
            p = path.point_from_proportion(alpha)
            dot.move_to(p)
            self.add(dot)

    def animate_flow(self, run_time=6, rate_func=linear):
        return [MoveAlongPath(d, self.path, run_time=run_time, rate_func=rate_func) for d in self]

# ---------------- SUBTITLE CHUNKS (full script split by scene) ----------------
SUBS = {
    "IntroScene": [
        "In the silence beneath our feet, a vast and intricate world hums with unseen life.",
        "A world older than forests, more complex than any computer network, and as essential to life on land as sunlight itself.",
        "This is the story of the mycorrhizal fungi — the subterranean architects of cooperation, the silent traders of life, the threads that bind the biosphere."
    ],
    "HistoryScene": [
        "For hundreds of millions of years, fungi and plants have coexisted in a partnership so deep that neither can be understood without the other.",
        "Long before animals took their first steps on land, fungal filaments were already weaving through the soil, forming microscopic bridges between the primitive roots of early plants.",
        "This alliance — the mycorrhizal symbiosis — allowed plants to conquer dry land. Fungi provided nutrients; plants offered sugars. An exchange of survival — evolution’s first great contract."
    ],
    "NetworkOverviewScene": [
        "Today, this ancient relationship persists beneath nearly every forest, grassland, and mountain slope.",
        "Mycorrhizal fungi extend far beyond the reach of roots, forming vast underground networks — delicate webs of hyphae that stretch through meters, even kilometers, of soil.",
        "Each strand, thinner than a human hair, acts as both an extension of the tree’s body and a bridge to others. Through these threads, a forest becomes a collective intelligence."
    ],
    "MiningExchangeScene": [
        "The fungi act as miners, extracting phosphates, nitrates, and micronutrients from soil particles inaccessible to plant roots.",
        "Their enzymes dissolve rock, liberating minerals that feed the tree’s growth. In return, the tree supplies carbohydrates — sugars crafted in its leaves.",
        "This flow of resources forms a dynamic feedback loop: trees reward fungi when light is plentiful; fungi redistribute nutrients under stress."
    ],
    "CommunicationScene": [
        "Through the mycorrhizal network, trees can sense and respond to the presence of others.",
        "Signals of stress, drought, or pest attack can travel underground, triggering defensive chemistry in distant trees.",
        "These exchanges are ecological strategy that gives rise to cooperation — a 'Wood Wide Web' of chemical messages."
    ],
    "ArchitectureScene": [
        "The name mycorrhiza — fungus and root — captures this union, but the term conceals a world of variation.",
        "Ectomycorrhizae form sheaths around roots, creating a Hartig net where nutrients are traded molecule by molecule.",
        "Arbuscular mycorrhizae penetrate root cells, forming arbuscules — tree-like interfaces inside plant tissue for intimate exchanges."
    ],
    "CompetitionScene": [
        "This relationship is not purely harmonious. Some plants, like certain orchids, exploit fungal networks without giving anything in return.",
        "Some fungi can manipulate partners, demanding more sugar than they return in nutrients — a biological tug-of-war between generosity and greed.",
        "Yet dynamic tension and feedback ensure the system tends toward stability over evolutionary time."
    ],
    "EcosystemsScene": [
        "In temperate forests the ectomycorrhizal networks dominate, linking oaks, beeches, and conifers into vast webs.",
        "In tropical ecosystems, arbuscular mycorrhizae connect dense vegetation and complex soils.",
        "Wherever plants endure — tundra, grassland, desert, urban park — fungi sculpt the invisible foundations of ecosystems."
    ],
    "ResearchMethodsScene": [
        "Scientists map these networks using isotopic tracers, DNA sequencing, and advanced imaging.",
        "Carbon fixed by one tree can appear in the tissues of another meters away, traveling through fungal threads.",
        "These methods reveal the forest behaves less like a collection of individuals and more like a distributed organism."
    ],
    "ThreatsRestorationScene": [
        "This ancient alliance faces unprecedented threats: deforestation severs fungal networks; tilling disrupts soil architecture; chemical fertilizers override natural symbioses.",
        "As soils warm and dry and biodiversity declines, these networks — silent and unseen — begin to fade.",
        "Restoration is possible: rewilding, regenerative agriculture, and inoculation with native fungi can revive these connections."
    ],
    "ConclusionScene": [
        "The story of mycorrhizal symbiosis is not a tale of the past — it is a vision of what sustains the future.",
        "Beneath every root tip and in every grain of soil, hyphae stretch onward — unseen, unheralded, indispensable.",
        "Heal the networks. Heal the planet."
    ]
}

# ---------------- SCENES ----------------
class IntroScene(Scene):
    def construct(self):
        # Title
        title = Text("The Hidden Network", font_size=96, weight=BOLD, color=WHITE)
        subtitle = Text("Mycorrhizal Symbiosis and the Secret Life of Trees", font_size=34, color="#cfe8d5")
        header = VGroup(title, subtitle).arrange(DOWN, buff=0.3).to_edge(UP, buff=0.8)

        # forest skyline (procedural)
        forest = VGroup()
        xs = np.linspace(-7, 7, 8)
        for x in xs:
            t = make_tree(size=random.uniform(0.9, 1.6))
            t.shift(RIGHT * x + DOWN * 0.6)
            forest.add(t)
        hyphae = Hyphae(start=DOWN * 2.1, depth=6, length=2.2, angle_spread=pi/2, color="#44b3b3", stroke_width=2.0)
        hyphae.shift(DOWN * 0.9)

        # invisible cubic path for particle demo
        path = CubicBezier(LEFT*0.7 + DOWN*0.6, LEFT*0.9 + DOWN*1.1, LEFT*1.5 + DOWN*1.5, LEFT*2.1 + DOWN*1.8)
        path.set_stroke(opacity=0)
        particles = NutrientParticles(path, count=10, color="#ffd166")

        self.play(FadeIn(header, shift=UP*0.4), run_time=1.2)
        self.play(LaggedStart(*[GrowFromCenter(t) for t in forest], lag_ratio=0.07), run_time=2.4)
        self.play(Create(hyphae, run_time=3.0))
        self.play(*particles.animate_flow(run_time=6), run_time=0.2)

        display_subtitles(self, SUBS["IntroScene"], SCENE_DURATIONS["IntroScene"])
        self.wait(1.0)

class HistoryScene(Scene):
    def construct(self):
        left_label = Text("Early plants", font_size=34, color="#e6f2ea").to_edge(LEFT).shift(UP*1.0)
        right_label = Text("Fungal hyphae", font_size=34, color="#e6f2ea").to_edge(RIGHT).shift(UP*1.0)
        early_plants = VGroup()
        for i in range(3):
            stem = Rectangle(width=0.08, height=0.9, fill_color="#9bbf6a", fill_opacity=1).move_to(LEFT*2.4 + RIGHT*(i*0.3) + DOWN*0.6)
            leaf = Ellipse(width=0.7, height=0.3, fill_color="#7fb05d", fill_opacity=1).next_to(stem, UP, buff=0.05)
            early_plants.add(stem, leaf)
        fungal_cluster = Hyphae(start=RIGHT*1.5 + DOWN*0.6, depth=5, length=1.1, angle_spread=pi/1.8, color="#48b2b2", stroke_width=2.2)
        arrow1 = Arrow(start=LEFT*1.2, end=RIGHT*0.6, buff=0.5, stroke_width=3.0).set_color("#ffd166")
        arrow2 = Arrow(start=RIGHT*0.6, end=LEFT*1.2, buff=0.5, stroke_width=3.0).set_color("#c9f0d5")
        self.play(FadeIn(left_label), FadeIn(right_label))
        self.play(LaggedStartMap(FadeIn, early_plants, shift=UP, lag_ratio=0.06), run_time=1.6)
        self.play(Create(fungal_cluster, run_time=2.6))
        # move arrows into view
        arrow1.move_to(DOWN * 0.25)
        arrow2.move_to(DOWN * 0.75)
        self.play(GrowArrow(arrow1), GrowArrow(arrow2))

        display_subtitles(self, SUBS["HistoryScene"], SCENE_DURATIONS["HistoryScene"])
        self.wait(1.0)

class NetworkOverviewScene(Scene):
    def construct(self):
        pts = [(-4, 1.2), (-2, 2.0), (0.5, 2.1), (2.8, 1.4), (-1.5, -0.8), (1.2, -1.5), (3.4, -0.6)]
        graph = VGroup()
        dots = []
        for x, y in pts:
            dot = Dot(point=np.array([x, y, 0]), radius=0.12, color="#8ee4af")
            label = Text("", font_size=20).next_to(dot, UP, buff=0.12)
            graph.add(dot, label)
            dots.append(dot)
        # edges
        for i in range(len(dots)):
            for j in range(i + 1, len(dots)):
                if random.random() < 0.6:
                    edge = Line(dots[i].get_center(), dots[j].get_center()).set_stroke("#56c1d0", width=2.0, opacity=0.95)
                    graph.add(edge)
        self.play(LaggedStartMap(FadeIn, graph, shift=UP, lag_ratio=0.08), run_time=2.2)
        display_subtitles(self, SUBS["NetworkOverviewScene"], SCENE_DURATIONS["NetworkOverviewScene"])
        # animate pulses on a subset of edges
        edges = [m for m in graph if isinstance(m, Line)]
        pulses = []
        for edge in np.random.choice(edges, size=min(7, len(edges)), replace=False):
            dot = Dot(radius=0.06, color="#ffd166").move_to(edge.get_start())
            self.add(dot)
            pulses.append(MoveAlongPath(dot, edge, run_time=3.6, rate_func=smooth))
        self.play(*pulses)
        self.wait(1.0)

class MiningExchangeScene(Scene):
    def construct(self):
        ground = Rectangle(width=14, height=3.6, fill_color="#2b1f11", fill_opacity=0.9).shift(DOWN*0.6)
        tree = make_tree(size=1.6).to_edge(UP).shift(LEFT*2.2)
        hypha = Hyphae(start=DOWN*1.5 + RIGHT*1.5, depth=6, length=1.3, angle_spread=pi/1.9, color="#3dbdbd", stroke_width=2.0)
        hypha.shift(RIGHT*0.8)
        self.play(FadeIn(ground), GrowFromCenter(tree))
        self.play(Create(hypha, run_time=2.6))
        p_icon = MathTex("P", font_size=64, color="#ffd166").move_to(hypha.get_center() + LEFT*0.8 + DOWN*0.4)
        n_icon = MathTex("N", font_size=64, color="#ffd166").move_to(hypha.get_center() + RIGHT*0.9 + DOWN*0.4)
        sugar = MathTex("C_{6}H_{12}O_{6}", font_size=28, color="#ffefc2").next_to(tree, DOWN, buff=0.1)
        self.play(FadeIn(p_icon), FadeIn(n_icon), FadeIn(sugar))
        p_path = Line(p_icon.get_center(), hypha.get_center() + LEFT*0.25)
        n_path = Line(n_icon.get_center(), hypha.get_center() + RIGHT*0.25)
        sugar_path = Line(sugar.get_center(), hypha.get_center())
        p_dot = Dot(radius=0.06, color="#ffd166").move_to(p_icon.get_center())
        n_dot = Dot(radius=0.06, color="#ffd166").move_to(n_icon.get_center())
        sugar_dot = Dot(radius=0.06, color="#ffefc2").move_to(sugar.get_center())
        self.play(MoveAlongPath(p_dot, p_path, run_time=2.2), MoveAlongPath(n_dot, n_path, run_time=2.2))
        self.play(MoveAlongPath(sugar_dot, sugar_path, run_time=3.0))
        display_subtitles(self, SUBS["MiningExchangeScene"], SCENE_DURATIONS["MiningExchangeScene"])
        self.wait(1.0)

class CommunicationScene(Scene):
    def construct(self):
        t1 = make_tree(size=1.0).move_to(LEFT*3 + DOWN*0.2)
        t2 = make_tree(size=1.3).move_to(ORIGIN + UP*0.2)
        t3 = make_tree(size=1.1).move_to(RIGHT*3 + DOWN*0.1)
        self.play(FadeIn(t1), FadeIn(t2), FadeIn(t3))
        root1 = Line(t1.get_bottom(), t2.get_bottom()).set_stroke("#56c1d0", width=3.0)
        root2 = Line(t2.get_bottom(), t3.get_bottom()).set_stroke("#56c1d0", width=3.0)
        self.play(Create(root1), Create(root2))
        # insect attack and alarm travelling
        self.play(t1.animate.set_color("#ff6b6b").scale(1.04), run_time=0.6)
        alarm = Dot(radius=0.07, color="#ff6b6b").move_to(t1.get_bottom())
        self.add(alarm)
        path = Line(alarm.get_center(), t3.get_bottom())
        self.play(MoveAlongPath(alarm, path, run_time=4.2, rate_func=linear))
        shield = Tex("✦", font_size=48, color="#f6f7d7").next_to(t3, UP, buff=0.1)
        self.play(FadeIn(shield), t3.animate.set_color("#fdfcdc"))
        display_subtitles(self, SUBS["CommunicationScene"], SCENE_DURATIONS["CommunicationScene"])
        self.wait(1.0)

class ArchitectureScene(Scene):
    def construct(self):
        root = Circle(radius=2.2, fill_color="#2f5b4b", fill_opacity=0.12, stroke_color="#2f5b4b")
        root_label = Text("Root Cross-Section", font_size=30, color="#e6f2ea").to_edge(UP)
        self.play(Write(root_label), Create(root))
        hartig = VGroup()
        for i in range(12):
            r = 1.6 - i * 0.12
            ring = Circle(radius=r, stroke_color="#49b2b2", stroke_width=1.0, fill_opacity=0)
            ring.shift(LEFT*0.3*random.uniform(-1,1) + DOWN*0.06*random.uniform(-1,1))
            hartig.add(ring)
        self.play(Create(hartig, run_time=2.0))
        cell = Circle(radius=0.75, fill_color="#2b6f6f", fill_opacity=0.12, stroke_color="#2b6f6f").shift(RIGHT*1.6 + DOWN*0.3)
        arbuscules = VGroup()
        for a in np.linspace(0, 2*pi, 10, endpoint=False):
            line = Line(cell.get_center(), cell.get_center() + 0.6 * np.array([np.cos(a), np.sin(a), 0]))
            line.set_stroke("#ff9f1c", width=2.0)
            arbuscules.add(line)
        self.play(Create(cell), Create(arbuscules))
        display_subtitles(self, SUBS["ArchitectureScene"], SCENE_DURATIONS["ArchitectureScene"])
        self.wait(1.0)

class CompetitionScene(Scene):
    def construct(self):
        network = Hyphae(start=DOWN*1.0, depth=5, length=1.4, angle_spread=pi/1.6, color="#4db6ac", stroke_width=2.0)
        self.play(Create(network, run_time=2.2))
        host = make_tree(size=1.3).move_to(LEFT*2 + DOWN*0.2)
        parasite = make_tree(size=0.9).move_to(RIGHT*2 + DOWN*0.2).set_color("#ff6b6b")
        parasite_label = Text("Orchid", font_size=26, color="#ffdede").next_to(parasite, RIGHT, buff=0.1)
        self.play(FadeIn(host), FadeIn(parasite), Write(parasite_label))
        host_dot = Dot(radius=0.06, color="#ffd166").move_to(host.get_bottom())
        path = Line(host.get_bottom(), parasite.get_bottom())
        self.play(MoveAlongPath(host_dot, path, run_time=3.6))
        valve = Tex("✕", font_size=48, color="#ff7b7b").move_to(network.get_center())
        self.play(FadeIn(valve), run_time=0.8)
        display_subtitles(self, SUBS["CompetitionScene"], SCENE_DURATIONS["CompetitionScene"])
        self.wait(1.0)

class EcosystemsScene(Scene):
    def construct(self):
        left_title = Text("Temperate forests", font_size=34, color="#e6f2ea").to_edge(LEFT).shift(UP*1.2)
        right_title = Text("Tropical soils", font_size=34, color="#e6f2ea").to_edge(RIGHT).shift(UP*1.2)
        left_circle = Circle(radius=2.0, color="#2f6b5a", fill_opacity=0.05).shift(LEFT*2.6)
        right_circle = Circle(radius=2.0, color="#b56a1a", fill_opacity=0.05).shift(RIGHT*2.6)
        left_h = Hyphae(start=left_circle.get_center() + DOWN*0.7, depth=6, length=1.05, angle_spread=pi/1.9, color="#2f8d8d")
        right_h = Hyphae(start=right_circle.get_center() + DOWN*0.7, depth=4, length=0.75, angle_spread=pi/1.2, color="#f2b46e")
        self.play(Write(left_title), Write(right_title))
        self.play(Create(left_circle), Create(right_circle))
        self.play(Create(left_h, run_time=2.2), Create(right_h, run_time=2.2))
        display_subtitles(self, SUBS["EcosystemsScene"], SCENE_DURATIONS["EcosystemsScene"])
        self.wait(1.0)

class ResearchMethodsScene(Scene):
    def construct(self):
        bench = Rectangle(width=12, height=2.0, fill_color="#545454", fill_opacity=0.55).to_edge(DOWN, buff=0.9)
        seq_label = Text("DNA Sequencing", font_size=30, color="#e6f2ea").to_edge(LEFT).shift(UP*0.6)
        iso_label = Text("Isotopic Tracing", font_size=30, color="#e6f2ea").to_edge(RIGHT).shift(UP*0.6)
        self.play(Create(bench), Write(seq_label), Write(iso_label))
        t1 = Dot(point=np.array([-3.2, 0.5, 0]), radius=0.12, color="#8ad3a0")
        t2 = Dot(point=np.array([3.2, 0.5, 0]), radius=0.12, color="#8ad3a0")
        net = Line(t1.get_center(), t2.get_center()).set_stroke("#56c1d0", width=3.0)
        self.play(FadeIn(t1), FadeIn(t2), Create(net))
        tracer = Dot(radius=0.07, color="#b388eb").move_to(t1.get_center())
        self.play(MoveAlongPath(tracer, net, run_time=5.6))
        display_subtitles(self, SUBS["ResearchMethodsScene"], SCENE_DURATIONS["ResearchMethodsScene"])
        self.wait(1.0)

class ThreatsRestorationScene(Scene):
    def construct(self):
        left_back = Rectangle(width=6, height=6, fill_color="#1f1f1f", fill_opacity=0.8).shift(LEFT*3.5)
        right_back = Rectangle(width=6, height=6, fill_color="#143d2e", fill_opacity=0.25).shift(RIGHT*3.5)
        left_trees = VGroup(*[Text("☠", font_size=48) for _ in range(3)]).arrange(DOWN, buff=0.2).move_to(left_back.get_center())
        right_trees = VGroup(*[Text("🌲", font_size=48) for _ in range(3)]).arrange(DOWN, buff=0.2).move_to(right_back.get_center())
        self.play(Create(left_back), Create(right_back))
        self.play(FadeIn(left_trees), FadeIn(right_trees))
        broken = VGroup()
        for i in range(4):
            ln = Line(LEFT*1.5 + DOWN*(i*0.35 - 0.6), RIGHT*1.5 + DOWN*(i*0.35 - 0.6))
            ln.set_stroke("#ff7373", width=6.0, opacity=0.9)
            broken.add(ln)
        broken.shift(LEFT*3.5)
        self.play(Create(broken))
        inoc = Dot(radius=0.07, color="#ffd166").move_to(right_back.get_center() + DOWN*1.2)
        self.play(FadeIn(inoc))
        new_hypha = Hyphae(start=right_back.get_center() + DOWN*1.2, depth=6, length=1.1, angle_spread=pi/1.8, color="#4db6ac", stroke_width=2.0)
        self.play(Create(new_hypha, run_time=3.0))
        display_subtitles(self, SUBS["ThreatsRestorationScene"], SCENE_DURATIONS["ThreatsRestorationScene"])
        self.wait(1.0)

class ConclusionScene(Scene):
    def construct(self):
        title = Text("Everything is connected.", font_size=72, color="#e6f2ea", weight=BOLD)
        subtitle = Text("Heal the networks. Heal the planet.", font_size=34, color="#cfe8d5")
        group = VGroup(title, subtitle).arrange(DOWN, buff=0.5)
        self.play(FadeIn(group, shift=UP*0.6), run_time=1.4)
        display_subtitles(self, SUBS["ConclusionScene"], SCENE_DURATIONS["ConclusionScene"])
        self.wait(1.0)
