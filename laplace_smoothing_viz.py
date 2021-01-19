#!/usr/bin/env python

from manimlib.imports import *
from scipy.stats import beta
# To watch one of these scenes, run the following:
# python -m manim example_scenes.py SquareToCircle -pl
#
# Use the flag -l for a faster rendering at a lower
# quality.
# Use -s to skip to the end and just save the final frame
# Use the -p to have the animation (or image, if -s was
# used) pop up once done.
# Use -n <number> to skip ahead to the n'th animation of a scene.
# Use -r <number> to specify a resolution (for example, -r 1080
# for a 1920x1080 video)

class PlotFunctions(GraphScene):
    CONFIG = {
        "x_min" : 0,
        "x_max" : 1.5,
        "y_min" : -1,
        "y_max" : 4,
        "x_axis_label": "$\\theta$",
        "y_axis_label": "",
        "graph_origin" : ORIGIN ,
        "function_color" : RED ,
        "axes_color" : BLUE,
        "x_labeled_nums" : [x/4. for x in range(0, 5, 1)],
        "x_label_decimals": 2
    }   
    def construct(self):
        set_background(self)
        self.setup_axes(animate=True)
        desc_text = TextMobject("""Uninformative 

Prior Distribution""")
        final_text = TextMobject("Posterior")
        final_text.to_edge(UP+RIGHT)
        final_text.set_color(BLACK)
        desc_text.to_edge(UP+RIGHT)
        desc_text.set_color(BLACK)
        text_count = TextMobject("$$P(\\theta | D)$$ $$E[\\theta] = {:.2f}$$ Heads: {}, Tails: {}".format(0.5, 0, 0))
        text_count.to_edge(DOWN+RIGHT)
        text_count.set_color(BLACK)
        
        func_graph=self.get_graph(self.get_beta_func(1,1),self.function_color)
        updates = [
            [2,1],
            [2,2],
            [3,2],
            [4,2],
            [5,2],
            [5,3],
            [5,4],
            [6,4],
            [7,4],
            [8,4],
            [8,5]
        ]
        text_count_updates = [TextMobject(
                "$$P(\\theta | D)$$ $$E[\\theta] = {:.2f}$$ Heads: {}, Tails: {}".format(
                    round((data[0])/(data[0] + data[1]), 2), data[0] - 1, data[1] - 1)
            ) for data in updates]
        for text_counter in text_count_updates:
            text_counter.to_edge(DOWN+RIGHT)
            text_counter.set_color(BLACK)
        mod_funcs = tuple([self.get_graph(self.get_beta_func(*args), self.function_color) for args in updates])
        self.play(ShowCreation(func_graph), run_time=1.5)
        self.play(ShowCreation(text_count), ShowCreation(desc_text), run_time=1.5)
        self.wait(2)
        self.play(FadeOut(desc_text))
        for i in range(len(updates)):
            self.play(Transform(func_graph, mod_funcs[i]),
                      Transform(text_count, text_count_updates[i]),
                      run_time=0.5)
            self.wait(0.5)
        self.wait(1)
        self.play(FadeIn(final_text))
        self.wait(3)
        self.play(FadeOut(final_text))
        self.wait(2)

    def get_beta_func(self, a, b):
        def f(x): return beta(a,b).pdf(x)
        return f