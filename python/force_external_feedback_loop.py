#!/usr/bin/env python3
"""
Generates the sum of 4 sine ways using the architecture in figure 1A.

File: force_external_feedback_loop.py

Copyright 2020 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""


import scipy.io
import scipy.sparse
import scipy.stats
import numpy
import math
import progressbar


class Sine4_1A():

    """
    4 sine waves using architecture of figure 1A
    """

    def __init__(self):
        """Initialise parameters"""
        print("Initialising network parameters and constants")
        self.seed = 4242
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

        # output weights
        self.wo = numpy.zeros((self.N, 1))
        # delta values
        self.dw = numpy.zeros((self.N, 1))
        # feedback connection weights
        self.normal_rng = numpy.random.default_rng(seed=self.seed)
        self.wf = 2.0 * (
            self.normal_rng.uniform(size=(self.N, 1))
            - 0.5
        )
        # weights for the recurrent network
        #  self.M = scipy.io.loadmat("M.mat")
        #  self.M = self.M['b']
        rvs = scipy.stats.norm(loc=0.0, scale=1.0).rvs
        self.M = scipy.sparse.random(
            self.N, self.N, density=self.p,
            format='csr',
            data_rvs=rvs
        ) * self.g * self.scale
        self.M = self.M.toarray()

        print("M is of type {}".format(type(self.M)))

        numpy.savetxt("M.csv", self.M, fmt='%.10f',
                      delimiter=',', newline='\n')

        # For stimulus signals
        self.amp = 1.3
        self.freq = 1.0/60.

        # Print all parameters
        print("{:<30}:{:<10} neurons".format("N", self.N))
        print("{:<30}:{:<10}".format("p", self.p))
        print("{:<30}:{:<10}".format("g", self.g))
        print("{:<30}:{:<10}".format("alpha", self.alpha))
        print("{:<30}:{:<10} seconds".format("nsecs", self.nsecs))
        print("{:<30}:{:<10} seconds".format("dt", self.dt))
        print("{:<30}:{:<10} dts".format("learn_every", self.learn_every))
        print("{:<30}:{:<10}".format("scale", self.scale))
        print("Network initialised.")

    def __train(self, traintime=None):
        """
        Train the system

        :traintime: length of training time (nsecs by default)
        :returns: nothing

        """
        if not traintime:
            traintime = self.nsecs
        print("Training network for {} seconds".format(traintime))

        # simulation time: training
        self.simtime_training = numpy.arange(0, traintime - self.dt, self.dt)

        # training signal
        self.ft = self.__setup_sin_signals(self.amp, self.freq,
                                           self.simtime_training)

        # hold values of z during the simulation
        self.zt = numpy.zeros(shape=(1, len(self.simtime_training)))
        # hold previous values of z during the simulation
        self.zpt = numpy.zeros(shape=(1, len(self.simtime_training)))
        # to hold magnitude of weight vector
        self.wo_mag = numpy.zeros(shape=(1, len(self.simtime_training)))

        # initial values
        x0 = numpy.random.default_rng().standard_normal(size=(self.N, 1)) * 0.5
        z0 = numpy.random.default_rng().standard_normal(size=(1)) * 0.5

        x = x0
        r = numpy.tanh(x)
        z = z0

        # progress bar
        widgets = [progressbar.Percentage(), progressbar.Bar(),
                   progressbar.FormatLabel(" Time elapsed: %(elapsed)s")]
        bar = progressbar.ProgressBar(widgets=widgets, max_value=100).start()
        ti = 0
        for t in self.simtime_training:
            # neuronal activity in recurrent network
            x = (
                (1.0-self.dt) * x +
                self.M.dot(r * self.dt) +
                self.wf * (z * self.dt)
            )
            # rates of neurons in the recurrent network
            r = numpy.tanh(x)
            # output signal
            z = numpy.dot(self.wo.transpose(), r)

            if ti % self.learn_every == 0:
                # update inverse correlation matrix
                self.P = self.__update_P(self.P, r)

                # Update weights
                self.wo = self.__update_w(self.wo, z, r, self.P, self.ft[ti])

            self.zt[0][ti] = z
            self.wo_mag[0][ti] = numpy.sqrt(
                numpy.dot(self.wo.transpose(), self.wo))

            # increment time
            ti += 1
            if ti % 10 == 0:
                bar.update(ti/len(self.simtime_training) * 100)

        bar.finish()
        error_avg = ((numpy.abs(self.zt - self.ft).sum())
                     / len(self.simtime_training))
        print("Training MAE: {}".format(error_avg))

        numpy.savetxt("ft.csv", self.ft.transpose(), fmt='%.10f',
                      delimiter='\t', newline='\n')
        numpy.savetxt("zt.csv", self.zt.transpose(), fmt='%.10f',
                      delimiter='\t', newline='\n')
        numpy.savetxt("wo_mag.csv", self.wo_mag.transpose(), fmt='%.10f',
                      delimiter='\t', newline='\n')

    def __update_P(self, P, r):
        """
        Update P matrix

        :P: P(t-delta t)
        :r: r(t)
        :returns: P(t)

        """
        k = P.dot(r)
        rPr = r.transpose().dot(k)
        c = 1.0/(1.0 + rPr)
        return (P - k.dot((k.transpose() * (c))))

    def __update_w(self, wo, z, r, P, ft):
        """
        Update weights

        :wo: current weights (w(t-delta t))
        :z: readout
        :r: output rates
        :ft: target function
        :returns: new weights (w(t))

        """
        # update the error for the linear readout
        e_minus = z - ft

        k = P.dot(r)
        rPr = r.transpose().dot(k)
        c = 1.0/(1.0 + rPr)

        # update the output weights
        # where is c coming from here? Not in equation 4
        dw = -e_minus * k * c
        return(wo + dw)

    def __test(self, testtime=None):
        """
        Test the system

        :testtime: length of testing phase (nsecs by default)

        :returns: nothing

        """
        if not testtime:
            testtime = self.nsecs
        print("Testing network for {} seconds".format(testtime))

        # simulation time: testing
        self.simtime_testing = numpy.arange(self.simtime_training,
                                            2 * self.nsecs - self.dt,
                                            self.dt)
        # testing signal
        self.ft2 = self.__setup_sin_signals(self.amp, self.freq,
                                            self.simtime_testing)

    def __setup_sin_signals(self,  amp, freq, simtime):
        """
        Set up sin signals

        :amp: amplitude
        :freq: frequency
        :simtime: simulation time to be set up for
        :returns: array list of values

        """
        ft = (
            (amp/1.0)*numpy.sin(1.0*math.pi*freq*simtime) +
            (amp/2.0)*numpy.sin(2.0*math.pi*freq*simtime) +
            (amp/3.0)*numpy.sin(4.0*math.pi*freq*simtime) +
            (amp/6.0)*numpy.sin(3.0*math.pi*freq*simtime)
        )
        ft /= 1.5
        return ft

    def runSim(self):
        """
        Run the simulation

        :returns: nothing

        """
        self.__train()


if __name__ == "__main__":
    sim = Sine4_1A()
    sim.runSim()
