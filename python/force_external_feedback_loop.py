#!/usr/bin/env python3
"""
Generates the sum of 4 sine ways using the architecture in figure 1A.

File: force_external_feedback_loop.py

Copyright 2020 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""


import scipy
import numpy


class Sine4_1A():

    """
    4 sine waves using architecture of figure 1A
    """

    def __init__(self):
        """Initialise parameters"""
        self.N = 1000  # number of neurons in recurrent network
        self.p = 0.1  # sparsity of connectivity
        self.g = 1.5  # g greater than 1 leads to chaotic networks.
        self.alpha = 1.0  # acts as learning rate; alpha << N
        self.nsecs = 1440  # sim length
        self.dt = 0.1  # time step
        self.learn_every = 2  # learning interval
        self.scale = pow(self.p * self.N, -0.5)  # ?
        self.P = numpy.identity(self.N) * 1/self.alpha  # initial value of P
        self.nRec2Out = self.N  # number of neurons projecting to output
        self.w = numpy.zeros((self.N, 1))  # output weights
        self.dw = numpy.zeros((self.N, 1))  # delta w
        self.wf = 2.0 * (
            numpy.random.default_rng().normal(size=(self.N, 1)) - 0.5
        )  # feedback connection weights

        # Print all parameters
        print("{:<30}:{:<10}".format("N", self.N))
        print("{:<30}:{:<10}".format("p", self.p))
        print("{:<30}:{:<10}".format("g", self.g))
        print("{:<30}:{:<10}".format("alpha", self.alpha))
        print("{:<30}:{:<10}".format("nsecs", self.nsecs))
        print("{:<30}:{:<10}".format("dt", self.dt))
        print("{:<30}:{:<10}".format("learn_every", self.learn_every))
        print("{:<30}:{:<10}".format("scale", self.scale))

    def setup(self):
        """
        Set up simulation

        :returns: TODO

        """
        # set up weights for the recurrent network
        self.M = scipy.sparse.random(
            self.N, self.N, density=self.p,
            random_state=numpy.random.RandomState.normal(seed=self.seed)
        ) * self.g * self.scale

    def runSim(self):
        """
        Run the simulation

        :returns: nothing

        """
        pass


if __name__ == "__main__":
    sim = Sine4_1A()
    sim.runSim()
