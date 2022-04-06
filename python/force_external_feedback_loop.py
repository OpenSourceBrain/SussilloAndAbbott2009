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
import time


class Sine4_1A():

    """
    4 sine waves using architecture of figure 1A
    """

    def __init__(self, network_scale=1):
        """
        Initialise parameters

        network_scale: integer: scale of the network
        """
        print("Initialising network parameters and constants")
        self.seed = 4242
        self.N = 1000 * network_scale  # number of neurons in recurrent network
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
        # can load a mat file
        #  self.M = scipy.io.loadmat("M.mat")
        #  self.M = self.M['b']
        # or generate it here
        rvs = scipy.stats.norm(loc=0.0, scale=1.0).rvs
        self.M = scipy.sparse.random(
            self.N, self.N, density=self.p,
            format='csr',
            data_rvs=rvs
        ) * self.g * self.scale
        self.M = self.M.toarray()

        numpy.savetxt("M.csv", self.M, fmt='%.10f',
                      delimiter=',', newline='\n')

        # For stimulus signals
        self.amp = 1.3
        self.freq = 1.0/60.

        # initial values
        self.x0 = (
            numpy.random.default_rng().standard_normal(size=(self.N, 1))
            * 0.5)
        self.z0 = (
            numpy.random.default_rng().standard_normal(size=(1))
            * 0.5)

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

    def train(self, traintime=None):
        """
        Train the system

        :traintime: length of training time (nsecs by default)
        :returns: nothing

        """
        if not traintime:
            traintime = self.nsecs

        # simulation time: training
        self.simtime_training = numpy.arange(self.dt, traintime + self.dt,
                                             self.dt)
        print("Training network for {} seconds in {} steps".format(
            self.simtime_training[-1] - self.simtime_training[0] + self.dt,
            len(self.simtime_training)))

        # training signal
        self.ft = self.__setup_sin_signals(self.amp, self.freq,
                                           self.simtime_training)

        # hold values of z during the simulation
        self.z_training = numpy.zeros(shape=(1, len(self.simtime_training)))
        # to hold magnitude of weight vector
        self.wo_mag = numpy.zeros(shape=(1, len(self.simtime_training)))

        # initialise
        self.x = self.x0
        r = numpy.tanh(self.x)
        self.z = self.z0

        # progress bar
        widgets = [progressbar.Percentage(), progressbar.Bar(),
                   progressbar.FormatLabel(" Time elapsed: %(elapsed)s")]
        bar = progressbar.ProgressBar(widgets=widgets, max_value=100).start()
        ti = 0
        for t in self.simtime_training:
            # neuronal activity in recurrent network
            self.x = (
                (1.0-self.dt) * self.x +
                self.M.dot(r * self.dt) +
                self.wf * (self.z * self.dt)
            )
            # rates of neurons in the recurrent network
            r = numpy.tanh(self.x)
            # output signal
            self.z = numpy.dot(self.wo.transpose(), r)

            if ti % self.learn_every == 0:
                # update inverse correlation matrix
                self.P = self.__update_P(self.P, r)

                # Update weights
                self.wo = self.__update_w(self.wo, self.z, r,
                                          self.P, self.ft[ti])

            self.z_training[0][ti] = self.z
            self.wo_mag[0][ti] = numpy.sqrt(
                numpy.dot(self.wo.transpose(), self.wo))

            # increment time
            ti += 1
            if ti % 10 == 0:
                bar.update(ti/len(self.simtime_training) * 100)

        bar.finish()
        error_avg = ((numpy.abs(self.z_training - self.ft).sum())
                     / len(self.simtime_training))
        print("Training MAE: {}".format(error_avg))

        numpy.savetxt("ft-train.csv", self.ft.transpose(), fmt='%.10f',
                      delimiter='\t', newline='\n')
        numpy.savetxt("zt-train.csv", self.z_training.transpose(), fmt='%.10f',
                      delimiter='\t', newline='\n')
        numpy.savetxt("wo_mag-train.csv", self.wo_mag.transpose(), fmt='%.10f',
                      delimiter='\t', newline='\n')

    def test(self, testtime=None):
        """
        Test the system

        :testtime: length of testing phase (nsecs by default)

        :returns: nothing

        """
        if not testtime:
            testtime = self.nsecs

        # simulation time: testing
        self.simtime_testing = numpy.arange(
            self.simtime_training[-1] + self.dt,
            2 * self.nsecs + self.dt,
            self.dt
        )
        print("Testing network for {} seconds in {} steps".format(
            self.simtime_testing[-1] - self.simtime_testing[0] + self.dt,
            len(self.simtime_testing)))

        # hold values of z during the simulation
        self.z_testing = numpy.zeros(shape=(1, len(self.simtime_testing)))
        # testing signal
        self.ft2 = self.__setup_sin_signals(self.amp, self.freq,
                                            self.simtime_testing)

        # do not need to be reinitialised
        #  self.x = self.x0
        #  self.z = self.z0
        r = numpy.tanh(self.x)

        # progress bar
        widgets = [progressbar.Percentage(), progressbar.Bar(),
                   progressbar.FormatLabel(" Time elapsed: %(elapsed)s")]
        bar = progressbar.ProgressBar(widgets=widgets, max_value=100).start()
        ti = 0
        for t in self.simtime_testing:
            # neuronal activity in recurrent network
            self.x = (
                (1.0-self.dt) * self.x +
                self.M.dot(r * self.dt) +
                self.wf * (self.z * self.dt)
            )

            # rates of neurons in the recurrent network
            r = numpy.tanh(self.x)
            # output signal
            self.z = numpy.dot(self.wo.transpose(), r)

            self.z_testing[0][ti] = self.z

            # increment time
            ti += 1
            if ti % 10 == 0:
                bar.update(ti/len(self.simtime_testing) * 100)

        error_avg = ((numpy.abs(self.z_testing - self.ft2).sum())
                     / len(self.simtime_testing))
        print("\nTesting MAE: {}".format(error_avg))

        numpy.savetxt("ft-test.csv", self.ft2.transpose(), fmt='%.10f',
                      delimiter='\t', newline='\n')
        numpy.savetxt("zt-test.csv", self.z_testing.transpose(), fmt='%.10f',
                      delimiter='\t', newline='\n')


if __name__ == "__main__":
    start = time.perf_counter()
    sim = Sine4_1A()
    sim.train()
    sim.test()
    stop = time.perf_counter()

    print(f"Took {stop - start} seconds to run")
