`import pictologics` takes about 0.3 s less (0.8 s instead of 1.1 s with a warm cache). The JIT warm-up no longer imports `scipy.signal` for an FFT convolution that no filter uses.
