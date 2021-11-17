#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: Sebastien M. Popoff
"""

import numpy as np
from .functions import cart2pol
from hexalattice.hexalattice import create_hex_grid
from functools import reduce

class IndexProfile():
    def __init__(self,npoints,areaSize, n_r = None, n_theta = None):
        '''
		Parameters
		----------
		
		npoints : int
			size in pixels of the area
			
		areaSize : float
			 size in um of the area (let it be larger that the core size!)
        '''
        self.npoints = npoints
        self.n = np.zeros([npoints]*2)
        self.areaSize = areaSize
        x = np.linspace(-areaSize/2,areaSize/2,npoints)#+1./(npoints-1)
        self.X,self.Y = np.meshgrid(x,x)
        self.TH, self.R = cart2pol(self.X, self.Y)
        self.dh = 1.*self.areaSize/(self.npoints-1.)
        self.radialFunc = None
        self.type =  None
	
    def initFromArray(self,n_array):
        assert(n_array.shape == self.n.shape)
        self.n = n_array
        self.NA = None
        self.radialFunc = None
        self.type = 'custom'
		
    def initFromRadialFunction(self, nr):
        self.radialFunc = nr
        self.n = np.fromiter((nr(i) for i in self.R.reshape([-1])), np.float32)
        
    def initParabolicGRIN(self,n1,a,NA):
        self.NA = NA
        self.a = a
        self.type = 'GRIN'
        n2 = np.sqrt(n1**2-NA**2)
        Delta = NA**2/(2.*n1**2)
		
        radialFunc = lambda r: np.sqrt(n1**2.*(1.-2.*(r/a)**2*Delta)) if r<a else n2
        
        self.initFromRadialFunction(radialFunc)
        
    def initStepIndex(self,n1,a,NA):
        self.NA = NA
        self.a = a
        self.type = 'SI'
        self.n1 = n1
        n2 = np.sqrt(n1**2-NA**2)
        #Delta = NA**2/(2.*n1**2)

        radialFunc = lambda r: n1 if r < a else n2

        self.initFromRadialFunction(radialFunc)

    def initStepIndexMultiCoreRadial(
            self, n1: float = 1.45, a: float = 1., delta: float = 0.039,
            dims: int = 4, layers: int = 1,
            NA: float = 0.4, core_pitch: float = 5,
            central_core_radius: float = 0):
        n2 = n1 * (1 - delta)
        self.NA = NA
        self.a = a
        self.type = 'SIMC'
        self.n1 = n1
        # A grid for layers
        lgrid = np.arange(layers + 1)
        # Angles for positions of cores in one layer
        pos_angles = [
            np.arange(0, 2 * np.pi, 2 * np.pi / dims / lnum) for lnum in lgrid]
        # Radius-vector modules of cores centrums
        pos_radiuses = lgrid * int((core_pitch + 2 * a) // self.dh)
        pos_radiuses[1:] += int((central_core_radius - a) // self.dh)
        # Coordinates of cores as all combinations of radiuses and angles
        cores_coords = [[
            [self.npoints // 2 + int(r * np.sin(t)),
             self.npoints // 2 + int(r * np.cos(t))]
            for t in _a] for r, _a in zip(pos_radiuses, pos_angles)]
        cores_coords = sum(cores_coords, [])
        cores_radiuses = [central_core_radius] + [a] * (len(cores_coords) - 1)

        self.n = np.ones_like(self.R) * n2
        for _a, indxs in zip(cores_radiuses, cores_coords):
            i, j = indxs
            self.n[np.sqrt((self.X - self.X[i, j]) ** 2 +
                   (self.Y - self.Y[i, j]) ** 2) < _a] = n1
        self.n.flatten()

    def initStepIndexMultiCoreHex(
            self, n1: float = 1.45, a: float = 1., delta: float = 0.039,
            NA: float = 0.4, core_pitch: float = 5,
            central_core_radius: float = 0,
            cladding_radius: float = 25):
        n2 = n1 * (1 - delta)
        self.NA = NA
        self.a = a
        self.type = 'SIMC'
        self.n1 = n1
        # A grid for layers
        n = cladding_radius / (a + core_pitch)
        hex_centers, _ = create_hex_grid(nx=n,
                                         ny=n,
                                         crop_circ=n / 2,
                                         )
        n = central_core_radius / (a + core_pitch)
        central_core_hex_centers, _ = create_hex_grid(nx=n,
                                                      ny=n,
                                                      crop_circ=n / 2,
                                                      )
        mask = reduce(
            lambda x, y: x | y,
            [
                np.array([np.allclose(x, y) for x in hex_centers])
                for y in central_core_hex_centers
            ])
        hex_centers = hex_centers[np.invert(mask)]
        cores_coords = ((2 * a + core_pitch) * hex_centers /
                        self.dh).astype(int) + self.npoints // 2
        self.n = np.ones_like(self.R) * n2
        for indxs in cores_coords:
            i, j = indxs
            self.n[(np.sqrt((self.X - self.X[i, j]) ** 2 +
                   (self.Y - self.Y[i, j]) ** 2) < a)] = n1

        self.n[(np.sqrt(self.X ** 2 + self.Y ** 2) < central_core_radius - self.dh)] = n1
        self.n[
            (np.sqrt(self.X ** 2 + self.Y ** 2) > central_core_radius - self.dh) &
            (np.sqrt(self.X ** 2 + self.Y ** 2) < central_core_radius + self.dh)] = n2

        self.n.flatten()

    # def initStepIndexMultiCoreHex(
    #         self, n1: float = 1.45, a: float = 1., delta: float = 0.039,
    #         layers: int = 1,
    #         NA: float = 0.4, core_pitch: float = 5,
    #         central_core_radius: float = 0):
    #     n2 = n1 * (1 - delta)
    #     self.NA = NA
    #     self.a = a
    #     self.type = 'SIMC'
    #     self.n1 = n1
    #     # A grid for layers
    #     dtheta = 2 * np.pi / 6
    #     lgrid = np.arange(layers + 1)
    #     # Angles for positions of cores in one layer
    #     pos_angles = [
    #         np.arange(
    #             0, 2 * np.pi, 2 * np.pi / (0 ** lnum + 6 * lnum))
    #         for lnum in lgrid + 1]
    #     # Radius-vector modules of cores centrums
    #     layers_radiuses = lgrid * int((core_pitch + 2 * a) // self.dh)
    #     layers_radiuses[1:] += int(
    #         (central_core_radius - a) / np.sin(dtheta) / self.dh)

    #     def pos_r(theta):
    #         return np.sin(dtheta) * np.sin(
    #             theta - dtheta * (-1 + int(theta / dtheta))) ** -1
    #     # Coordinates of cores as all combinations of radiuses and angles
    #     cores_coords = [[
    #         [self.npoints // 2 + int(pos_r(t) * _lr * np.sin(t)),
    #          self.npoints // 2 + int(pos_r(t) * _lr * np.cos(t))]
    #         for t in _a] for _lr, _a in zip(layers_radiuses, pos_angles)]

    #     cores_coords = sum(cores_coords, [])
    #     cores_radiuses = [central_core_radius] + [a] * (len(cores_coords) - 1)
    #     self.n = np.ones_like(self.R) * n2
    #     for _a, indxs in zip(cores_radiuses, cores_coords):
    #         i, j = indxs
    #         self.n[(np.sqrt((self.X - self.X[i, j]) ** 2 +
    #                (self.Y - self.Y[i, j]) ** 2) < _a)] = n1
    #     self.n.flatten()
        
    def initStepIndexConcentric(
            self, n1: float = 1.45, a: float = 1., delta: float = 0.039,
            NA: float = 0.4, core_pitch: float = 5, layers: int = 1):
        n2 = n1 * (1 - delta)
        self.NA = NA
        self.a = a
        self.type = 'SIC'
        self.n1 = n1
        radiuses = np.arange(layers + 1) * core_pitch

        def radialFunc(r):
            for r0 in radiuses:
                if np.abs(r - r0) < a:
                    return n1
            return n2

        self.initFromRadialFunction(radialFunc)

    def initPhotonicCrystalRadial(
            self, n1: float = 1.45, a: float = 1.,
            dims: int = 4, layers: int = 1,
            NA: float = 0.4, core_pitch: float = 5,
            cladding_radius: float = 25,
            central_core_radius: float = 0):
        delta = 1 - n1
        n2 = np.sqrt(n1**2-NA**2)
        self.initStepIndexMultiCore(
            1, a, delta, dims, layers, NA, core_pitch,
            central_core_radius)
        self.type = 'PCF'
        self._crop_cladding(n2, cladding_radius)

    def initPhotonicCrystalHex(
            self, n1: float = 1.45, a: float = 1.,
            NA: float = 0.4, core_pitch: float = 5,
            pcf_radius=24,
            cladding_radius: float = 65,
            central_core_radius: float = 0):
        delta = 1 - n1
        n2 = np.sqrt(n1**2-NA**2)
        self.initStepIndexMultiCoreHex(
            1, a, delta, NA, core_pitch,
            central_core_radius, pcf_radius)
        self.type = 'PCF'
        self._crop_cladding(n2, cladding_radius)

    def _crop_cladding(self, n2, cladding_radius):
        self.n[self.R > cladding_radius] = n2
