#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Grace Li (li85)
Date: 8/11/2021

Grains class to represent approximated analytic solution of a shock through 
a grid of grains. The solution is composed from a truncated Fourier series
of single mode density perturbations from the the equations in the 
Velikovich et al. 2007 paper: 
''Shock front distortion and Richtmyer-Meshkov-type growth caused by 
a small preshock nonuniformity''
Phys. Plasmas 14, 072706 (2007)

"""

# Required Imports
# import time
import math
import numpy as np
from numpy import pi as pi
import warnings

from . import SingleMode
# from SingleMode import SingleModeSolution

class GrainsSolution:
    
    def __init__(self, 
                 Lg_x, Lg_y, Ls, Lt,
                 rho_g, rho_s, 
                 M1,
                 rho1_reference = 3.5,
                 tol = None, 
                 n_x_modes = None, n_y_modes = None,
                 max_modes = 5000,
                 D = 2.39, gamma = 5/3):
        '''
        Constructs the necessary attributes for the Grains object.

        Parameters
        ----------
        Lg_x : float [nm]
            Length of grains in the x-direction.
            The unit can be specified to be your choice, as long as
            the lengths Lg_x, Lg_y, Ls, and Lt have the same units.
            The x and y coordinates will also be in this length unit.
        Lg_y : float [nm]
            Length of grains in the y-direction
            The unit can be specified to be your choice, as long as
            the lengths Lg_x, Lg_y, Ls, and Lt have the same units.
            The x and y coordinates will also be in this length unit.
        Ls : float [nm]
            Length of interstitial spaces. 
            This is the same in the x and y directions.
            The unit can be specified to be your choice, as long as
            the lengths Lg_x, Lg_y, Ls, and Lt have the same units.
            The x and y coordinates will also be in this length unit.
        Lt : float [nm]
            Length of a single transition region between grain and interstitial space.
            In a single period, there are two transition regions: one going 
            from a grain to an interstitial space, and one going from an
            interstitial space to a grain.
            The unit can be specified to be your choice, as long as
            the lengths Lg_x, Lg_y, Ls, and Lt have the same units.
            The x and y coordinates will also be in this length unit.
        rho_g : float [g/cm3]
            Density of the grains.
        rho_s : float [g/cm3]
            Density of the interstitial spaces
        M1 : float [dimensionless]
            Mach number of the incident shock.
            M1 = D/a1 where D is the shock speed if the preshock fluid is at rest,
            and a1 is the speed of sound in the preshock fluid.
        rho1_reference : float [g/cm3], default = 3.5
            Reference density that perturbations should be compared to. All 
            perturbations will be compared to if a pre-shock region of uniform 
            density equal to rho_reference.
        
        Additional Parameters
        ---------------------
        tol : float, optional
            Error tolerance allowed between the truncated Fourier series and the
            actual density perturbation.
            If tol, n_xmodes, and n_ymodes are not provided, then the this defaults
            to 0.1 * (rho_g - rho_s), which is 10% of the density difference.
            If only one of n_xmodes and n_ymodes are provided, the default tol is used.
        n_x_modes : int, optional
            The number of Fourier modes in the x direction to represent the 
            density perturbation. If tol is provided, this is ignored.
        n_y_modes : int, optional
            The number of Fourier modes in the y direction to represent the 
            density perturbation. If tol is provided, this is ignored.
        max_modes: int, default = 2000
            The maximum number of Fourier modes in either direction
        
        D : float [cm / microsecond], default = 2.39
            Shock speed (relative to the lab reference frame) in cm / microsecond,
            which is also 10 * km/second
        gamma : float, default = 5/3
            Adiabatic index (also known as heat capacity ratio).
            This analysis uses gamma in the ideal gas equation of state.

        Returns
        -------
        A fully constructed instance of the Grains object.

        '''
        
        ## Initialize preshock region parameters
        self.M1 = M1
        self.gamma = gamma
        self.D = D
        
        ## Initialize Length scales
        self.Lg_x = Lg_x
        self.Lg_y = Lg_y
        self.Lt = Lt
        self.Ls = Ls
        
        ## Calculate the period in x and y
        self.Lx = Lg_x + Ls + 2*Lt
        self.Ly = Lg_y + Ls + 2*Lt
        
        ## Initialize densities and calculate baseline mean density rho1,
        #  which depends on the Fourier coefficients of the constant mode
        self.rho_g = rho_g
        self.rho_s = rho_s
        self.rho1_reference = rho1_reference
        
        ## Initialize max number of Fourier modes
        self.max_modes = max_modes
        
        ## Calculate the Fourier coefficients for the constant mode in the
        # x and y directions
        high = math.sqrt(self.rho_g - self.rho_s)
        low = 0
        self.cx0 = (self.Lg_x*high + self.Ls*low + self.Lt*(high+low)) / self.Lx
        self.cy0 = (self.Lg_y*high + self.Ls*low + self.Lt*(high+low)) / self.Ly
        
        self.rho1 = self.rho_s + self.cx0 * self.cy0
        
        #Calculate the preshock speed of sound
        self.a1 = D/self.M1
        
        #Calculate the preshock pressure form an ideal gas EOS
        #The units are [g / (cm * microseconds^2)]
        self.p1 = self.rho1 * (self.a1)**2 / self.gamma 
        
        # Calculate average post shock quantities
        self.rho2 = ( (gamma+1)*(M1**2) / ((gamma-1)*(M1**2) + 2) )* self.rho1
        self.rho2_reference = ( (gamma+1)*(M1**2) / ((gamma-1)*(M1**2) + 2) )* self.rho1_reference
        self.p2 = ((2*gamma*(M1**2) - gamma + 1) / (gamma + 1) )* self.p1
        a2 = math.sqrt( (2*gamma*(M1**2) - gamma + 1) * ((gamma-1)*(M1**2) + 2) )
        self.a2 = ( a2 / ( (gamma+1)*M1 ) )* self.a1
        M2 = ( (gamma-1)*(M1**2) + 2 ) / (2*gamma*(M1**2) - gamma + 1)
        self.M2 = math.sqrt(M2)
        
        #Calculate the post shock fluid speed relative to the preshock fluid
        #To be used as the lab reference frame velocity for normalization
        #U = Difference in velocity of pre and post shock fluids
        self.U = self.D - self.M2*self.a2
        
        ## Vectorize density functions with conditional statements to take numpy input
        self.rho_x = np.vectorize(self.rho_x)
        self.rho_y = np.vectorize(self.rho_y)
        # self.x_mode = np.vectorize(self.x_mode)
        # self.y_mode = np.vectorize(self.y_mode)
        
        ## Determine the number of Fourier modes in the x and y directions
        #If tol is provided, calculate the number of modes using tol
        if tol != None:
            self.get_n_modes(tol)
        #If tol is not provided, but both of n_xmodes and n_ymodes are provided, use them
        elif (n_x_modes != None) and (n_y_modes != None):
            self.n_x_modes = n_x_modes 
            self.n_y_modes = n_y_modes
        #Otherwise use the default tol and calculate the number of modes
        else:
            tol = 0.1 * (self.rho_g - self.rho_s)
            self.get_n_modes(tol)
        
        #Calculate and store Fourier coefficients
        self.get_fourier_coeff()
        
    ##-----------------------------------------------------------------------
    ## Define x and y components of density. These will be multiplied and shifted
    # to give the actual density perturbation profile in 2D. We use these x and y
    # density components do Fourier series decomposition in each direction.
    
    def rho_x(self, x):
        '''
        Shifted x-component contribution to the density.
        
        Parameters
        ----------
        x : float or np.array of floats 
            x-coordinate(s) to evaluate at.

        Returns
        -------
        float or np.array with shape equal to x.shape
            Shifted x-component contribution to the density.
        
        To match the setup in the Velikovich et al. the density is represented
        as an even periodic function so the middle of the grain starts at x=0.
        The transitions between grain to interstitial space are cubic smooth-step
        functions which match the density values and zero-derivative condition
        at the grain and interstitial space interfaces.
        
        This density is shifted to set the density in the interstitial space 
        regions equal to zero to allow combination by multiplying with the 
        y-component of density. The grain density is set to sqrt(rho_g - rho_s) 
        also to allow for combination with the y-component density.
        '''
        
        #Set the high and low values of the smooth-step curve
        high = math.sqrt(self.rho_g - self.rho_s)
        low = 0
        d = high - low
        
        #Make function definition periodic with period Lx
        while x > self.Lx:
            x = x - self.Lx
        while x < 0:
            x = x + self.Lx
        
        #Return the value of the piecewise continuous grain-step function        
        #First part of a grain, with no density pertu  rbation
        if x <= self.Lg_x/2:
            return high
        #First transition region from grain to intersitial space
        elif x < self.Lg_x/2 + self.Lt:
            density = 2*d * ((x - self.Lg_x/2)/self.Lt)**3
            density = density - 3*d * ((x - self.Lg_x/2)/self.Lt)**2
            density = density + high
            return density
        #Interstitial step which negative density perturbation
        elif x <= self.Lg_x/2 + self.Lt + self.Ls:
            return low
        #Second transition region from interstitial step to grain
        elif x < self.Lg_x/2 + 2*self.Lt + self.Ls:
            density = -2*d * ((x - (self.Lg_x/2 + self.Ls + self.Lt))/self.Lt)**3
            density = density + 3*d * ((x - (self.Lg_x/2 + self.Ls + self.Lt))/self.Lt)**2
            density = density + low
            return density
        #Second part of the grain
        elif x <= self.Lx:
            return high
        
        #Because we defined the function to be periodic, we should not need this case
        #If we have some bad input however, return 0 for no density perturbation
        else:
            error = "Function input was not valid and no density perturbation will be returned"
            warnings.warn(error, Warning)
            return 0
        
    def rho_y(self, y):
        '''
        Shifted y-component contribution to the density.
        
        Parameters
        ----------
        y : float or np.array of floats 
            y-coordinate(s) to evaluate at.

        Returns
        -------
        float or np.array with shape equal to y.shape
            Shifted y-component contribution to the density.
            
        To match the setup in the Velikovich et al. the density is represented
        as an even periodic function so the middle of the grain starts at y=0.
        The transitions between grain to interstitial space are cubic smooth-step
        functions which match the density values and zero-derivative condition
        at the grain and interstitial space interfaces.
            
        This density is shifted to set the density in the interstitial space 
        regions equal to zero to allow combination by multiplying with the 
        y-component of density. The grain density is set to sqrt(rho_g - rho_s) 
        also to allow for combination with the y-component density.
        '''
        
        #Set the high and low values of the smooth-step curve
        high = math.sqrt(self.rho_g - self.rho_s)
        low = 0
        d = high - low
        
        #Make function definition periodic with period Ly
        while y > self.Ly:
            y = y - self.Ly
        while y < 0:
            y = y + self.Ly
            
        #Return the value of the piecewise continuous grain-step function        
        if y <= self.Lg_y/2:
            return high
        elif y < self.Lg_y/2 + self.Lt:
            density = 2*d * ((y - self.Lg_y/2)/self.Lt)**3
            density = density - 3*d * ((y - self.Lg_y/2)/self.Lt)**2
            density = density + high
            return density
        elif y <= self.Lg_y/2 + self.Lt + self.Ls:
            return low
        elif y < self.Lg_y/2 + 2*self.Lt + self.Ls:
            density = -2*d * ((y - (self.Lg_y/2 + self.Ls + self.Lt))/self.Lt)**3
            density = density + 3*d * ((y - (self.Lg_y/2 + self.Ls + self.Lt))/self.Lt)**2
            density = density + low
            return density
        elif y <= self.Ly:
            return high
        #Because we defined the function to be periodic, we should not need this case
        #If we have some bad input however, return 0 for no density perturbation
        else:
            error = "Function input was not valid and no density perturbation will be returned"
            warnings.warn(error, Warning)
            return 0
    
    ##-----------------------------------------------------------------------
    ## Define analytically calculated Fourier coefficients for a single mode
    # and functions to represent a single mode in the x and y directions
    
    # Define Fourier coefficents in x-direction
    def c_x(self, n):
        '''
        Fourier coefficient for the nth mode in the x-direction.
        
        Parameters
        ----------
        n : int >= 1
            Which Fourier mode to calculate the coefficient for in the x-direction.
            We must have n >= 1. For n = 0, use the value self.cx0 instead.

        Returns
        -------
        float
            Returns the Fourier coefficient for the nth mode in the x-direction. 
            
        The Fourier coefficient is calculated analytically for the function 
        form of rho_x with cubic smooth-step transition regions. 
        For other density functional forms, a numerical implementation 
        to determine the Fourier coefficients may be implemented instead.
        '''
        
        if 0 in np.array(n):
            raise ValueError('Function only accepts integer modes n >= 1')
        
        high = math.sqrt(self.rho_g - self.rho_s)
        low = 0
        d = high - low
        
        ## Calculate the Fourier coefficient for the n >= 1 modes with cosine component
        #Calculate the frequency
        freq = 2*pi * n / self.Lx
        
        #Coefficient contribution from grain
        coeff = np.sin(freq*self.Lg_x/2) - np.sin(freq*(self.Lx - self.Lg_x/2))
        coeff = high * (1/freq) * coeff
        
        #Add coefficient contribution from interstitial space
        coeff = coeff + low * 2 * (1/freq) * np.sin(freq * self.Ls/2) * np.cos(n*pi/2)
        
        #Add contribution from the two transition regions
        a = self.Lg_x/2+ self.Lt
        b = self.Lg_x/2 + self.Ls + 2*self.Lt
        
        part1 = self.Lt * freq * ((self.Lt*freq)**2 - 6) * (np.sin(freq*a) - np.sin(freq*b))
        part1 = part1 + 3 * ((self.Lt*freq)**2 - 2) * (np.cos(freq*a) - np.cos(freq*b))
        part1 = part1 + 6 * ( np.cos(freq*(a-self.Lt)) - np.cos(freq*(b-self.Lt)) )
        part1 = 2 * d / ( (self.Lt**3) * (freq**4) ) * part1
        
        part2  = ((self.Lt*freq)**2 - 2) * (np.sin(freq*a) - np.sin(freq*b))
        part2 = part2 + 2*self.Lt*freq * (np.cos(freq*a) - np.cos(freq*b))
        part2 = part2 + 2 * ( np.sin(freq*(a-self.Lt)) - np.sin(freq*(b-self.Lt)) )
        part2 = -3 * d / ((self.Lt**2) * (freq**3)) * part2

        part3 = high * ( np.sin(freq*a) - np.sin(freq*(a-self.Lt)) )
        part3 = part3 + low * ( np.sin(freq*b) - np.sin(freq*(b-self.Lt)) )
        part3 = part3 / freq
        
        coeff = coeff + part1 + part2 + part3
        coeff = coeff * 2 / self.Lx
        return coeff

    # Define single fourier mode in x-direction
    def x_mode(self, x, n):
        '''
        nth Fourier mode in the x-direction for n >= 1.
        
        Parameters
        ----------
        x : float or np.array of floats
            x-coordinates to evaluate the nth x-direction Fourier mode at.
        n : int >= 1
            Which Fourier mode in the x-direction. 
            This function only takes n >= 1.

        Returns
        -------
        float or np.array of floats with same shape as x
            Returns the value at x of the nth Fourier mode in the x-direction.
        '''
        if n == 0:
            try:
                return self.cx0 * np.ones(x.shape)
            except:
                return self.cx0
        
        return self.c_x(n) * np.cos(2*pi * n * x / self.Lx)
        
    # Define Fourier coefficents in y-direction
    def c_y(self, n):
        '''
        Fourier coefficient for the nth mode in the y-direction.
        
        Parameters
        ----------
        n : int >= 1
            Which Fourier mode to calculate the coefficient for in the y-direction.
            We must have n >= 1. For n = 0, use the value self.cy0 instead.

        Returns
        -------
        float
            Returns the Fourier coefficient for the nth mode in the y-direction. 
            
        The Fourier coefficient is calculated analytically for the function 
        form of rho_x with cubic smooth-step transition regions. 
        For other density functional forms, a numerical implementation 
        to determine the Fourier coefficients may be implemented instead.
        '''
        
        if 0 in np.array(n):
            raise ValueError('Function only accepts integer modes n >= 1')
        
        high = math.sqrt(self.rho_g - self.rho_s)
        low = 0
        d = high - low
        
        ##Calculate the Fourier coefficient for the n >= 1 modes with cosine component
        
        #Calculate the frequency
        freq = 2*pi * n / self.Ly
        
        #Coefficient contribution from grain
        coeff = np.sin(freq*self.Lg_y/2) - np.sin(freq*(self.Ly - self.Lg_y/2))
        coeff = high * (1/freq) * coeff
        
        #Add coefficient contribution from interstitial space
        coeff = coeff + low * 2 * (1/freq) * np.sin(freq * self.Ls/2) * np.cos(n*pi/2)
        
        #Add contribution from the two transition regions
        a = self.Lg_y/2+ self.Lt
        b = self.Lg_y/2 + self.Ls + 2*self.Lt
        
        part1 = self.Lt * freq * ((self.Lt*freq)**2 - 6) * (np.sin(freq*a) - np.sin(freq*b))
        part1 = part1 + 3 * ((self.Lt*freq)**2 - 2) * (np.cos(freq*a) - np.cos(freq*b))
        part1 = part1 + 6 * ( np.cos(freq*(a-self.Lt)) - np.cos(freq*(b-self.Lt)) )
        part1 = 2 * d / ( (self.Lt**3) * (freq**4) ) * part1
        
        part2  = ((self.Lt*freq)**2 - 2) * (np.sin(freq*a) - np.sin(freq*b))
        part2 = part2 + 2*self.Lt*freq * (np.cos(freq*a) - np.cos(freq*b))
        part2 = part2 + 2 * ( np.sin(freq*(a-self.Lt)) - np.sin(freq*(b-self.Lt)) )
        part2 = -3 * d / ((self.Lt**2) * (freq**3)) * part2

        part3 = high * ( np.sin(freq*a) - np.sin(freq*(a-self.Lt)) )
        part3 = part3 + low * ( np.sin(freq*b) - np.sin(freq*(b-self.Lt)) )
        part3 = part3 / freq
        
        coeff = coeff + part1 + part2 + part3
        coeff = coeff * 2 / self.Ly
        return coeff
    
    # Define single fourier mode in y-direction
    def y_mode(self, y, n):
        '''
        nth Fourier mode in the y-direction for n >= 1.
        
        Parameters
        ----------
        y : float or np.array of floats
            y-coordinates to evaluate the nth y-direction Fourier mode at.
        n : int >= 1
            Which Fourier mode in the y-direction. 
            This function only takes n >= 1.

        Returns
        -------
        float or np.array of floats with same shape as y
            Returns the value at y of the nth Fourier mode in the y-direction.
        '''
        if n == 0:
            try:
                return self.cy0 * np.ones(y.shape)
            except:
                return self.cy0
        
        return self.c_y(n) * np.cos(2*pi * n * y / self.Ly)
        
    ##-----------------------------------------------------------------------
    # Define 2-D density perturbation function
    def preshock_delta_rho(self, x, y, perturb_from_mean = False):
        '''
        Preshock density perturbation from baseline density at x and y.
        By default, perturb_from_mean is False, and the density perturbation
        is calculated with respect to rho1_reference, the provided reference density.
        If instead, perturb_from_mean is set to True, then the perturbation will
        be calcualted with respect to the mean preshock density rho1.
        '''
        #Recall the baseline density rho1 = cx0*cy0 + rho_s
        #so really we're multiplying the x and y density contributions 
        #and subtracting the constant Fourier mode 
        density_from_mean = self.rho_x(x) * self.rho_y(y) - (self.rho1 - self.rho_s)
        if perturb_from_mean:
            return density_from_mean
        else:
            return density_from_mean + (self.rho1 - self.rho1_reference)
       
    ##-----------------------------------------------------------------------
    # Calculate number of x and y modes for particular error bound tol (if provided)
    def get_n_modes(self, tol):
        '''
        Calculates and sets the number of Fourier modes in the x and y directions,
        n_x_modes and n_y_modes needed to represent the density perturbation
        within some specified global tolerance tol. The number of modes calculated
        overwrites any provided in the initial object construction.
        
        This funciton works if tol is provided in the initial class construction 
        or this method is run on it's own with an input tol value.
        '''
        
        # Calculate the required 1D tolerance in each of x and y 
        # to gurantee a global 2D tolerance tol
        high = math.sqrt(self.rho_g - self.rho_s)
        tol_1D = math.sqrt(high**2 + tol) - high
        
        ## Get number of modes required for x-direction
        #The largest errors will be near the transition region, so we check there
        x = np.linspace(self.Lg_x/2 * 0.9, self.Lx - (self.Lg_x/2 * 0.9), 5000)
        #Compute the actual density perturbation values
        actual_delta_rho = self.rho_x(x)
        
        #Calculate truncated Fourier series and keep adding terms until tol_1D is reached
        Fx = self.cx0 * np.ones(x.size)
        self.n_x_modes = self.max_modes
        for i in range(1, self.max_modes):
            Fx = Fx + self.x_mode(x, i)
            difference = abs(Fx - actual_delta_rho)
            if max(difference) < tol_1D:
                self.n_x_modes = i
                break
        
        #If the max_nodes was reached in the x-direction, warn the user
        if self.n_x_modes == self.max_modes:
            error = "The maximum number of modes (" + str(self.max_modes) 
            error = error + ") was reached in the x-direction"
            warnings.warn(error, Warning)
            
        
        ## Get number of modes required for y-direction
        #The largest errors will be near the transition region, so we check there
        y = np.linspace(self.Lg_y/2 * 0.9, self.Ly - (self.Lg_y/2 * 0.9), 5000)
        #Compute the actual density perturbation values
        actual_delta_rho = self.rho_y(y)
        
        #Calculate truncated Fourier series and keep adding terms until tol_1D is reached
        Fy = self.cy0 * np.ones(y.size)
        self.n_y_modes = self.max_modes
        for i in range(1, self.max_modes):
            Fy = Fy + self.y_mode(y, i)
            difference = abs(Fy - actual_delta_rho)
            if max(difference) < tol_1D:
                self.n_y_modes = i
                break
        
        #If the max_nodes was reached in the y-direction, warn the user
        if self.n_y_modes == self.max_modes:
            error = "The maximum number of modes (" + str(self.max_modes) 
            error = error + ") was reached in the y-direction"
            warnings.warn(error, Warning)
    
    ##-----------------------------------------------------------------------
    #Once the number of modes is set, we can calculate and store the Fourier
    #coeffients define a function for the Fourier representation of density
    def get_fourier_coeff(self):
        '''
        Calculate and store as a numpy row array the Fourier coefficients of the 
        x and y direction up to n_x_modes and n_y_modes.
        '''  
        x_coeff = np.array(range(1, self.n_x_modes))
        self.x_coeff = np.insert(self.c_x(x_coeff), 0, self.cx0)
        
        y_coeff = np.array(range(1, self.n_y_modes))
        self.y_coeff = np.insert(self.c_y(y_coeff), 0, self.cy0)
    
    def fourier_delta_rho(self, x, y, perturb_from_mean = False):
        '''
        Fourier approximation of density perturbations.
        Takes position x, y and returns the truncated Fourier series
        representation of the density perturbation at that point or at
        all points in a grid of the x and y points if they're arrays.
        
        By default, perturb_from_mean is False, and the density perturbation
        is calculated with respect to rho1_reference, the provided reference density.
        If instead, perturb_from_mean is set to True, then the perturbation will
        be calcualted with respect to the mean preshock density rho1.
        '''
        
        #Calculate truncated Fourier series in the x direction
        try:
            Fx = self.cx0 * np.ones(x.shape)
        except:
            Fx = self.cx0
        for i in range(1, self.n_x_modes):
            Fx = Fx + self.x_mode(x, i)
    
        #Calculate truncated Fourier series in the y direction
        try:
            Fy = self.cy0 * np.ones(y.shape)
        except:
            Fy = self.cy0
        for i in range(1, self.n_y_modes):
            Fy = Fy + self.y_mode(y, i)
            
        #Return the outer product to get the density at all points in the x-y grid
        #Since we want the perturbation, subtract out the c_x(0) * c_y(0) constant
        perturbation_from_mean = np.outer(Fy, Fx) - self.cx0 * self.cy0
        if perturb_from_mean:
            return perturbation_from_mean
        else:
            return perturbation_from_mean + (self.rho1 - self.rho1_reference)
    
    ##-----------------------------------------------------------------------
    # # Get the dimensional version of perturbations that come from direct
    # linear combination of the Fourier modes. These are the pressure (sonic),
    # density (sonic, entropy), and x and y velocity (sonic, vortex) perturbations.
    
    def tilde_rho_s(self, t, x, y, perturb_from_mean = False):
        '''
        Dimensionless sonic density perturbation. 
        Normalized by rho2 or rho2_reference.

        Parameters
        ----------
        t : float [microsecond]
            Time coordinate.
        x : float or np.array of floats [nm]
            x coordinates(s). Must be the same shape as y.
            Must be the same length unit as the grain and interstitial widths.
        y : float or np.array of floats [nm]
            y coordinates(s). Must be the same shape as x.
            Must be the same length unit as the grain and interstitial widths.
        perturb_from_mean : bool, default = False
            If perturb_from_mean is False (the default), the the density perturbation
            is calculated with respect to rho2_reference, from the provided reference 
            density. If perturn_from_mean is True, then the perturbation is calculated
            with respect to the mean density rho2.

        Returns
        -------
        float or np.array of floats with same shape as x and y
            Returns the value of the dimensionless sonic density perturbation
            at x and y, or at all pairs of coordinates (x,y) if np.array of 
            floats are given. The dimensionless quantity is normalized by rho2
            or rho2_reference
        '''
        #Start with baseline perturbation of zero
        try:
            perturbation = np.zeros(x.shape)
        except:
            perturbation = 0
        
        #Add the perturbations from all mode combos other than the 0,0 constant mode
        for i in range(self.n_x_modes):
            for j in range(self.n_y_modes):
                if not (i==0 and j==0):
                
                    #Calculate normalized density perturbation amplitude
                    epsilon_k = self.x_coeff[i] * self.y_coeff[j] / self.rho1
                
                    sim = SingleMode.SingleModeSolution(
                            Nx = i, Ny = j, M1 = self.M1, 
                            Lx = self.Lx, Ly = self.Ly,
                            epsilon_k = epsilon_k,
                            rho1 = self.rho1, 
                            D = self.D,
                            gamma = self.gamma)
                    
                    perturbation = perturbation + np.real(sim.tilde_rho_s(t,x,y))
        
        if perturb_from_mean:
            return perturbation
        else:
            perturbation = self.rho2 * perturbation + (self.rho2 - self.rho2_reference)
            return perturbation / self.rho2_reference
    
    def tilde_rho_e(self, t, x, y, perturb_from_mean = False):
        '''
        Dimensionless entropy density perturbation. 
        Normalized by rho2 or rho2_reference.

        Parameters
        ----------
        t : float [microsecond]
            Time coordinate.
        x : float or np.array of floats [nm]
            x coordinates(s). Must be the same shape as y.
            Must be the same length unit as the grain and interstitial widths.
        y : float or np.array of floats [nm]
            y coordinates(s). Must be the same shape as x.
            Must be the same length unit as the grain and interstitial widths.
        perturb_from_mean : bool, default = False
            If perturb_from_mean is False (the default), the the density perturbation
            is calculated with respect to rho2_reference, from the provided reference 
            density. If perturn_from_mean is True, then the perturbation is calculated
            with respect to the mean density rho2.

        Returns
        -------
        float or np.array of floats with same shape as x and y
            Returns the value of the dimensionless entropy density perturbation
            at x and y, or at all pairs of coordinates (x,y) if np.array of 
            floats are given. The dimensionless quantity is normalized by rho2
            or rho2_reference.
        '''
        
        #Start with baseline perturbation of zero
        try:
            perturbation = np.zeros(x.shape)
        except:
            perturbation = 0
        
        #Add the perturbations from all mode combos other than the 0,0 constant mode
        for i in range(self.n_x_modes):
            for j in range(self.n_y_modes):
                if not (i==0 and j==0):
                
                    #Calculate normalized density perturbation amplitude
                    epsilon_k = self.x_coeff[i] * self.y_coeff[j] / self.rho1
                
                    sim = SingleMode.SingleModeSolution(
                            Nx = i, Ny = j, M1 = self.M1, 
                            Lx = self.Lx, Ly = self.Ly,
                            epsilon_k = epsilon_k,
                            rho1 = self.rho1, 
                            D = self.D,
                            gamma = self.gamma)
                    
                    perturbation = perturbation + np.real(sim.tilde_rho_e(t,x,y))
        
        if perturb_from_mean:
            return perturbation
        else:
            perturbation = self.rho2 * perturbation + (self.rho2 - self.rho2_reference)
            return perturbation / self.rho2_reference
    
    def tilde_rho(self, t, x, y, perturb_from_mean = False):
        '''
        Dimensionless total (sonic + entropy) density perturbation.
        Normalized by rho2 or rho2_reference.

        Parameters
        ----------
        t : float [microsecond]
            Time coordinate.
        x : float or np.array of floats [nm]
            x coordinates(s). Must be the same shape as y.
            Must be the same length unit as the grain and interstitial widths.
        y : float or np.array of floats [nm]
            y coordinates(s). Must be the same shape as x.
            Must be the same length unit as the grain and interstitial widths.
        perturb_from_mean : bool, default = False
            If perturb_from_mean is False (the default), the the density perturbation
            is calculated with respect to rho2_reference, from the provided reference 
            density. If perturn_from_mean is True, then the perturbation is calculated
            with respect to the mean density rho2.

        Returns
        -------
        float or np.array of floats with same shape as x and y
            Returns the value of the dimensionless total (sonic + entropy) 
            density perturbation at x and y, or at all pairs of coordinates (x,y)
            if np.array of floats are given. The dimensionless quantity is
            normalized by rho2 or rho2_reference.
        '''
        
        #Start with baseline perturbation of zero
        try:
            perturbation = np.zeros(x.shape)
        except:
            perturbation = 0
        
        #Add the perturbations from all mode combos other than the 0,0 constant mode
        for i in range(self.n_x_modes):
            for j in range(self.n_y_modes):
                if not (i==0 and j==0):
                
                    #Calculate normalized density perturbation amplitude
                    epsilon_k = self.x_coeff[i] * self.y_coeff[j] / self.rho1
                
                    sim = SingleMode.SingleModeSolution(
                            Nx = i, Ny = j, M1 = self.M1, 
                            Lx = self.Lx, Ly = self.Ly,
                            epsilon_k = epsilon_k,
                            rho1 = self.rho1, 
                            D = self.D,
                            gamma = self.gamma)
                    
                    perturbation = perturbation + np.real(sim.tilde_rho(t,x,y))
        
        if perturb_from_mean:
            return perturbation
        else:
            perturbation = self.rho2 * perturbation + (self.rho2 - self.rho2_reference)
            return perturbation / self.rho2_reference
    
    def tilde_p(self, t, x, y):
        '''
        Dimensionless total pressure perturbation. Normalized by p2.
        This consists of sonic component only.
        
        Parameters
        ----------
        t : float [microsecond]
            Time coordinate.
        x : float or np.array of floats [nm]
            x coordinates(s). Must be the same shape as y.
            Must be the same length unit as the grain and interstitial widths.
        y : float or np.array of floats [nm]
            y coordinates(s). Must be the same shape as x.
            Must be the same length unit as the grain and interstitial widths.

        Returns
        -------
        float or np.array of floats with same shape as x and y
            Returns the value of the dimensionless total (sonic component only)
            pressure perturbation at x and y, or at all pairs of coordinates (x,y)
            if np.array of floats are given. The dimensionless quantity is
            normalized by p2.
        '''
        
        #Start with baseline perturbation of zero
        try:
            perturbation = np.zeros(x.shape)
        except:
            perturbation = 0
        
        #Add the perturbations from all mode combos other than the 0,0 constant mode
        for i in range(self.n_x_modes):
            for j in range(self.n_y_modes):
                if not (i==0 and j==0):
                    
                    #Calculate normalized density perturbation amplitude
                    epsilon_k = self.x_coeff[i] * self.y_coeff[j] / self.rho1
                
                    sim = SingleMode.SingleModeSolution(
                            Nx = i, Ny = j, M1 = self.M1, 
                            Lx = self.Lx, Ly = self.Ly,
                            epsilon_k = epsilon_k,
                            rho1 = self.rho1, 
                            D = self.D,
                            gamma = self.gamma)
                    
                    perturbation = perturbation + np.real(sim.tilde_p(t,x,y))
                    
        return perturbation

    def tilde_vx_s(self, t, x, y):
        '''
        Dimensionless sonic velocity perturbation in x-component. 
        Normalized by a2.
        
        Parameters
        ----------
        t : float [microsecond]
            Time coordinate.
        x : float or np.array of floats [nm]
            x coordinates(s). Must be the same shape as y.
            Must be the same length unit as the grain and interstitial widths.
        y : float or np.array of floats [nm]
            y coordinates(s). Must be the same shape as x.
            Must be the same length unit as the grain and interstitial widths.

        Returns
        -------
        float or np.array of floats with same shape as x and y
            Returns the value of the dimensionless sonic x-velocity perturbation 
            at x and y, or at all pairs of coordinates (x,y) if np.array of 
            floats are given. The dimensionless quantity is normalized by a2.
            The velocity perturbation is from 0 since we are in the reference
            frame where the postshock region is stationary.
        '''
        
        #Start with baseline perturbation of zero
        try:
            perturbation = np.zeros(x.shape)
        except:
            perturbation = 0
        
        #Add the perturbations from all mode combos other than the 0,0 constant mode
        for i in range(self.n_x_modes):
            for j in range(self.n_y_modes):
                if not (i==0 and j==0):
                    
                    #Calculate normalized density perturbation amplitude
                    epsilon_k = self.x_coeff[i] * self.y_coeff[j] / self.rho1
                
                    sim = SingleMode.SingleModeSolution(
                            Nx = i, Ny = j, M1 = self.M1, 
                            Lx = self.Lx, Ly = self.Ly,
                            epsilon_k = epsilon_k,
                            rho1 = self.rho1, 
                            D = self.D,
                            gamma = self.gamma)
                    
                    perturbation = perturbation + np.real(sim.tilde_vx_s(t,x,y))
                    
        return perturbation
    
    def tilde_vx_v(self, t, x, y):
        '''
        Dimensionless vortex velocity perturbation in x-component. 
        Normzlized by a2.
        
        Parameters
        ----------
        t : float [microsecond]
            Time coordinate.
        x : float or np.array of floats [nm]
            x coordinates(s). Must be the same shape as y.
            Must be the same length unit as the grain and interstitial widths.
        y : float or np.array of floats [nm]
            y coordinates(s). Must be the same shape as x.
            Must be the same length unit as the grain and interstitial widths.

        Returns
        -------
        float or np.array of floats with same shape as x and y
            Returns the value of the dimensionless vortex x-velocity perturbation 
            at x and y, or at all pairs of coordinates (x,y) if np.array of 
            floats are given. The dimensionless quantity is normalized by a2.
            The velocity perturbation is from 0 since we are in the reference
            frame where the postshock region is stationary.
        '''
        
        #Start with baseline perturbation of zero
        try:
            perturbation = np.zeros(x.shape)
        except:
            perturbation = 0
        
        #Add the perturbations from all mode combos other than the 0,0 constant mode
        for i in range(self.n_x_modes):
            for j in range(self.n_y_modes):
                if not (i==0 and j==0):
                    
                    #Calculate normalized density perturbation amplitude
                    epsilon_k = self.x_coeff[i] * self.y_coeff[j] / self.rho1
                
                    sim = SingleMode.SingleModeSolution(
                            Nx = i, Ny = j, M1 = self.M1, 
                            Lx = self.Lx, Ly = self.Ly,
                            epsilon_k = epsilon_k,
                            rho1 = self.rho1, 
                            D = self.D,
                            gamma = self.gamma)
                    
                    perturbation = perturbation + np.real(sim.tilde_vx_v(t,x,y))
                    
        return perturbation
    
    def tilde_vx(self, t, x, y):
        '''
        Dimensionless total (sonic + vortex) velocity perturbation in x-component. 
        Normalized by a2.
        
        Parameters
        ----------
        t : float [microsecond]
            Time coordinate.
        x : float or np.array of floats [nm]
            x coordinates(s). Must be the same shape as y.
            Must be the same length unit as the grain and interstitial widths.
        y : float or np.array of floats [nm]
            y coordinates(s). Must be the same shape as x.
            Must be the same length unit as the grain and interstitial widths.

        Returns
        -------
        float or np.array of floats with same shape as x and y
            Returns the value of the dimensionless total (sonic + vortex) 
            x-velocity perturbation at x and y, or at all pairs of coordinates 
            (x,y) if np.array of floats are given. The dimensionless quantity 
            is normalized by a2. The velocity perturbation is from 0 since we 
            are in the reference frame where the postshock region is stationary.
        '''
        
        #Start with baseline perturbation of zero
        try:
            perturbation = np.zeros(x.shape)
        except:
            perturbation = 0
        
        #Add the perturbations from all mode combos other than the 0,0 constant mode
        for i in range(self.n_x_modes):
            for j in range(self.n_y_modes):
                if not (i==0 and j==0):
                    
                    #Calculate normalized density perturbation amplitude
                    epsilon_k = self.x_coeff[i] * self.y_coeff[j] / self.rho1
                
                    sim = SingleMode.SingleModeSolution(
                            Nx = i, Ny = j, M1 = self.M1, 
                            Lx = self.Lx, Ly = self.Ly,
                            epsilon_k = epsilon_k,
                            rho1 = self.rho1, 
                            D = self.D,
                            gamma = self.gamma)
                    
                    perturbation = perturbation + np.real(sim.tilde_vx(t,x,y))
                    
        return perturbation
    
    def tilde_vy_s(self, t, x, y):
        '''
        Dimensionless sonic velocity perturbation in y-component.
        Normalized by a2.
        
        Parameters
        ----------
        t : float [microsecond]
            Time coordinate.
        x : float or np.array of floats [nm]
            x coordinates(s). Must be the same shape as y.
            Must be the same length unit as the grain and interstitial widths.
        y : float or np.array of floats [nm]
            y coordinates(s). Must be the same shape as x.
            Must be the same length unit as the grain and interstitial widths.

        Returns
        -------
        float or np.array of floats with same shape as x and y
            Returns the value of the dimensionless sonic y-velocity perturbation 
            at x and y, or at all pairs of coordinates (x,y) if np.array of 
            floats are given. The dimensionless quantity is normalized by a2.
        '''
        
        #Start with baseline perturbation of zero
        try:
            perturbation = np.zeros(x.shape)
        except:
            perturbation = 0
        
        #Add the perturbations from all mode combos other than the 0,0 constant mode
        for i in range(self.n_x_modes):
            for j in range(self.n_y_modes):
                if not (i==0 and j==0):
                    
                    #Calculate normalized density perturbation amplitude
                    epsilon_k = self.x_coeff[i] * self.y_coeff[j] / self.rho1
                
                    sim = SingleMode.SingleModeSolution(
                            Nx = i, Ny = j, M1 = self.M1, 
                            Lx = self.Lx, Ly = self.Ly,
                            epsilon_k = epsilon_k,
                            rho1 = self.rho1, 
                            D = self.D,
                            gamma = self.gamma)
                    
                    perturbation = perturbation + np.real(sim.tilde_vy_s(t,x,y))
                    
        return perturbation
    
    def tilde_vy_v(self, t, x, y):
        '''
        Dimensionless vortex velocity perturbation in y-component.
        Normalized by a2.
        
        Parameters
        ----------
        t : float [microsecond]
            Time coordinate.
        x : float or np.array of floats [nm]
            x coordinates(s). Must be the same shape as y.
            Must be the same length unit as the grain and interstitial widths.
        y : float or np.array of floats [nm]
            y coordinates(s). Must be the same shape as x.
            Must be the same length unit as the grain and interstitial widths.

        Returns
        -------
        float or np.array of floats with same shape as x and y
            Returns the value of the dimensionless vortex y-velocity perturbation 
            at x and y, or at all pairs of coordinates (x,y) if np.array of 
            floats are given. The dimensionless quantity is normalized by a2.
        '''
        #Start with baseline perturbation of zero
        try:
            perturbation = np.zeros(x.shape)
        except:
            perturbation = 0
        
        #Add the perturbations from all mode combos other than the 0,0 constant mode
        for i in range(self.n_x_modes):
            for j in range(self.n_y_modes):
                if not (i==0 and j==0):
                    
                    #Calculate normalized density perturbation amplitude
                    epsilon_k = self.x_coeff[i] * self.y_coeff[j] / self.rho1
                
                    sim = SingleMode.SingleModeSolution(
                            Nx = i, Ny = j, M1 = self.M1, 
                            Lx = self.Lx, Ly = self.Ly,
                            epsilon_k = epsilon_k,
                            rho1 = self.rho1, 
                            D = self.D,
                            gamma = self.gamma)
                    
                    perturbation = perturbation + np.real(sim.tilde_vy_v(t,x,y))
                    
        return perturbation
    
    def tilde_vy(self, t, x, y):
        '''
        Dimensionless total (sonic + vortex) velocity perturbation in y-component
        Normalized by a2.
        
        Parameters
        ----------
        t : float [microsecond]
            Time coordinate.
        x : float or np.array of floats [nm]
            x coordinates(s). Must be the same shape as y.
            Must be the same length unit as the grain and interstitial widths.
        y : float or np.array of floats [nm]
            y coordinates(s). Must be the same shape as x.
            Must be the same length unit as the grain and interstitial widths.

        Returns
        -------
        float or np.array of floats with same shape as x and y
            Returns the value of the dimensionless total (sonic + vortex) 
            y-velocity perturbation at x and y, or at all pairs of coordinates 
            (x,y) if np.array of floats are given. The dimensionless quantity 
            is normalized by a2.
        '''
        #Start with baseline perturbation of zero
        try:
            perturbation = np.zeros(x.shape)
        except:
            perturbation = 0
        
        #Add the perturbations from all mode combos other than the 0,0 constant mode
        for i in range(self.n_x_modes):
            for j in range(self.n_y_modes):
                if not (i==0 and j==0):
                    
                    #Calculate normalized density perturbation amplitude
                    epsilon_k = self.x_coeff[i] * self.y_coeff[j] / self.rho1
                
                    sim = SingleMode.SingleModeSolution(
                            Nx = i, Ny = j, M1 = self.M1, 
                            Lx = self.Lx, Ly = self.Ly,
                            epsilon_k = epsilon_k,
                            rho1 = self.rho1, 
                            D = self.D,
                            gamma = self.gamma)
                    
                    perturbation = perturbation + np.real(sim.tilde_vy(t,x,y))
                    
        return perturbation
    
    ##-----------------------------------------------------------------------
    ## Get the dimensional (delta) version of perturbations.
    ## These quantities must be real, physical values
    ## Relations between dimensional (delta) and nondimensional (tilde)
    ## values comes from Eqn 5 in the paper
    
    def delta_rho(self, t, x, y, component = 'total', perturb_from_mean = False):
        '''
        Dimensional density perturbation in units of [g / cm3]
        Component can be specified to be 'sonic', or 'entropy' to just
        give that solution component. By default returns the total perturbation.
        
        Parameters
        ----------
        t : float [microsecond]
            Time coordinate.
        x : float or np.array of floats [nm]
            x coordinates(s). Must be the same shape as y.
            Must be the same length unit as the grain and interstitial widths.
        y : float or np.array of floats [nm]
            y coordinates(s). Must be the same shape as x.
            Must be the same length unit as the grain and interstitial widths.
        component: string, default = 'total'
            Which solution component to return the density perturbation of.
            The options are 'sonic', 'entropy', and 'total' (the default)
        perturb_from_mean : bool, default = False
            If perturb_from_mean is False (the default), the the density perturbation
            is calculated with respect to rho2_reference, from the provided reference 
            density. If perturn_from_mean is True, then the perturbation is calculated
            with respect to the mean density rho2.

        Returns
        -------
        float or np.array of floats with same shape as x and y
            Returns the value of the density perturbation in units of [g / cm3]
            of the specified component at x and y, or at all pairs of coordinates 
            (x,y) if np.array of floats are given. Density perturbation is
            calculated with respect to rho2 or rho2_reference depending on the 
            value of perturb_from_mean.
        
        '''
        if component in ['sonic', 's']:
            if perturb_from_mean:
                return self.rho2 * self.tilde_rho_s(t,x,y, perturb_from_mean)
            else:
                return self.rho2_reference * self.tilde_rho_s(t,x,y)
        elif component in ['entropy', 'e']:
            if perturb_from_mean:
                return self.rho2 * self.tilde_rho_e(t,x,y, perturb_from_mean)
            else:
                return self.rho2_reference * self.tilde_rho_e(t,x,y)
        else:
            if perturb_from_mean:
                return self.rho2 * self.tilde_rho(t,x,y, perturb_from_mean)
            else:
                return self.rho2_reference * self.tilde_rho(t,x,y)
    
    def delta_p(self, t, x, y):
        '''
        Dimensional total (sonic only) pressure perturbation
        in units of [g / (cm * microseconds^2)]
        
        Parameters
        ----------
        t : float [microsecond]
            Time coordinate.
        x : float or np.array of floats [nm]
            x coordinates(s). Must be the same shape as y.
            Must be the same length unit as the grain and interstitial widths.
        y : float or np.array of floats [nm]
            y coordinates(s). Must be the same shape as x.
            Must be the same length unit as the grain and interstitial widths.

        Returns
        -------
        float or np.array of floats with same shape as x and y
            Returns the value of the pressure perturbation in units of
            [g / (cm * microseconds^2)] at x and y, or at all pairs of coordinates 
            (x,y) if np.array of floats are given.
        
        '''
        return self.gamma * self.p2 * self.tilde_p(t,x,y)
     
    def delta_vx(self, t, x, y, component = 'total'):
        '''
        Dimensional velocity perturbation in x-component
        in units of [cm / microsecond].
        Component can be specified to be 'sonic', or 'vortex' to just
        give that solution component. By default returns the total perturbation.
        
        Parameters
        ----------
        t : float [microsecond]
            Time coordinate.
        x : float or np.array of floats [nm]
            x coordinates(s). Must be the same shape as y.
            Must be the same length unit as the grain and interstitial widths.
        y : float or np.array of floats [nm]
            y coordinates(s). Must be the same shape as x.
            Must be the same length unit as the grain and interstitial widths.
        component: string, default = 'total'
            Which solution component to return the density perturbation of.
            The options are 'sonic', 'vortex', and 'total' (the default)

        Returns
        -------
        float or np.array of floats with same shape as x and y
            Returns the value of the x-velocity perturbation in units of 
            [cm / microsecond] of the specified component at x and y, or at 
            all pairs of coordinates (x,y) if np.array of floats are given. 
            The velocity perturbation is from 0 since we are in the reference
            frame where the postshock region is stationary.
    
        '''
        
        if component in ['sonic', 's']:
            return self.a2 * self.tilde_vx_s(t,x,y)
        elif component in ['vortex', 'v']:
            return self.a2 * self.tilde_vx_v(t,x,y)
        else:
            return self.a2 * self.tilde_vx(t,x,y)
    
    def delta_vy(self, t, x, y, component = 'total'):
        '''
        Dimensional velocity perturbation in x-component
        in units of [cm / microsecond].
        Component can be specified to be 'sonic', or 'vortex' to just
        give that solution component. By default returns the total perturbation.
        
        Parameters
        ----------
        t : float [microsecond]
            Time coordinate.
        x : float or np.array of floats [nm]
            x coordinates(s). Must be the same shape as y.
            Must be the same length unit as the grain and interstitial widths.
        y : float or np.array of floats [nm]
            y coordinates(s). Must be the same shape as x.
            Must be the same length unit as the grain and interstitial widths.
        component: string, default = 'total'
            Which solution component to return the density perturbation of.
            The options are 'sonic', 'vortex', and 'total' (the default)

        Returns
        -------
        float or np.array of floats with same shape as x and y
            Returns the value of the y-velocity perturbation in units of 
            [cm / microsecond] of the specified component at x and y, or at 
            all pairs of coordinates (x,y) if np.array of floats are given. 
        '''
        if component in ['sonic', 's']:
            return self.a2 * self.tilde_vy_s(t,x,y)
        elif component in ['vortex', 'v']:
            return self.a2 * self.tilde_vy_v(t,x,y)
        else:
            return self.a2 * self.tilde_vy(t,x,y)
    
    ##-----------------------------------------------------------------------
    ## Define the combined shock front perturbation
    def delta_xs(self, t, y):
        '''
        Perturbation of the shock front in the x-direction at the given time t
        for a given (float or np.array) y-coordinate(s).
        
        Parameters
        ----------
        t : float [microsecond]
            Time coordinate.
        y : float or np.array of floats [nm]
            y coordinates(s).
            Must be the same length unit as the grain and interstitial widths.

        Returns
        -------
        float or np.array of floats with same shape as y
            Returns the value of the perturbation of the shock front 
            in the x-direction at the given time t for given (float or np.array) 
            y-coordinate(s). The perturbation is in units of [nm], 
            or the same unit as the grain/interstitial spaces
        '''
        #Start with baseline perturbation of zero
        try:
            perturbation = np.zeros(y.shape)
        except:
            perturbation = 0
        
        #Add the perturbations from all mode combos other than the 0,0 constant mode
        for i in range(self.n_x_modes):
            for j in range(self.n_y_modes):
                if not (i==0 and j==0):
                    
                    #Calculate normalized density perturbation amplitude
                    epsilon_k = self.x_coeff[i] * self.y_coeff[j] / self.rho1
                
                    sim = SingleMode.SingleModeSolution(
                            Nx = i, Ny = j, M1 = self.M1, 
                            Lx = self.Lx, Ly = self.Ly,
                            epsilon_k = epsilon_k,
                            rho1 = self.rho1, 
                            D = self.D,
                            gamma = self.gamma)
                    
                    perturbation = perturbation + np.real(sim.delta_xs(t,y))
                    
        return perturbation
    
    ##-----------------------------------------------------------------------
    ## Define divergence and curl - both normalized and dimensional
    
    def div_tilde_v(self, t, x, y):
        '''
        Normalized divergence of the perturbed velocity. Normalized by a2.
        
        Parameters
        ----------
        t : float [microsecond]
            Time coordinate.
        x : float or np.array of floats [nm]
            x coordinates(s). Must be the same shape as y.
            Must be the same length unit as the grain and interstitial widths.
        y : float or np.array of floats [nm]
            y coordinates(s). Must be the same shape as x.
            Must be the same length unit as the grain and interstitial widths.

        Returns
        -------
        float or np.array of floats with same shape as x and y
            Returns the value of the divergence of the dimensionless (total)
            perturbation in velocity of the specified component at x and y, 
            or at all pairs of coordinates (x,y) if np.array of floats are given.
            Quantity is normalized by a2.
        '''
        #Start with baseline perturbation of zero
        try:
            perturbation = np.zeros(x.shape)
        except:
            perturbation = 0
        
        #Add the perturbations from all mode combos other than the 0,0 constant mode
        for i in range(self.n_x_modes):
            for j in range(self.n_y_modes):
                if not (i==0 and j==0):
                    
                    #Calculate normalized density perturbation amplitude
                    epsilon_k = self.x_coeff[i] * self.y_coeff[j] / self.rho1
                
                    sim = SingleMode.SingleModeSolution(
                            Nx = i, Ny = j, M1 = self.M1, 
                            Lx = self.Lx, Ly = self.Ly,
                            epsilon_k = epsilon_k,
                            rho1 = self.rho1, 
                            D = self.D,
                            gamma = self.gamma)
                    
                    perturbation = perturbation + np.real(sim.div_tilde_v(t,x,y))
                    
        return perturbation
    
    def div_delta_v(self, t, x, y):
        '''
        Dimensional divergence of the velocity perturbation in 
        in units of [1 / microsecond].
        
        Parameters
        ----------
        t : float [microsecond]
            Time coordinate.
        x : float or np.array of floats [nm]
            x coordinates(s). Must be the same shape as y.
            Must be the same length unit as the grain and interstitial widths.
        y : float or np.array of floats [nm]
            y coordinates(s). Must be the same shape as x.
            Must be the same length unit as the grain and interstitial widths.

        Returns
        -------
        float or np.array of floats with same shape as x and y
            Returns the value of the divergence of the total velocity perturbation
            in units of [1 / microsecond] of the specified component at x and y, 
            or at all pairs of coordinates (x,y) if np.array of floats are given. 
        '''
        return self.a2 * self.div_tilde_v(t, x, y)
    
    def curl_tilde_v(self, t, x, y):
        '''
        Normalized perturbed vorticity (curl of the perturbed velocity).
        Normalized by a2.
        
        Parameters
        ----------
        t : float [microsecond]
            Time coordinate.
        x : float or np.array of floats [nm]
            x coordinates(s). Must be the same shape as y.
            Must be the same length unit as the grain and interstitial widths.
        y : float or np.array of floats [nm]
            y coordinates(s). Must be the same shape as x.
            Must be the same length unit as the grain and interstitial widths.

        Returns
        -------
        float or np.array of floats with same shape as x and y
            Returns the value of the curl of the dimensionless (total)
            perturbation in velocity of the specified component at x and y, 
            or at all pairs of coordinates (x,y) if np.array of floats are given. 
            Quantity is normalized by a2.
        '''
    
        #Start with baseline perturbation of zero
        try:
            perturbation = np.zeros(x.shape)
        except:
            perturbation = 0
        
        #Add the perturbations from all mode combos other than the 0,0 constant mode
        for i in range(self.n_x_modes):
            for j in range(self.n_y_modes):
                if not (i==0 and j==0):
                    
                    #Calculate normalized density perturbation amplitude
                    epsilon_k = self.x_coeff[i] * self.y_coeff[j] / self.rho1
                
                    sim = SingleMode.SingleModeSolution(
                            Nx = i, Ny = j, M1 = self.M1, 
                            Lx = self.Lx, Ly = self.Ly,
                            epsilon_k = epsilon_k,
                            rho1 = self.rho1, 
                            D = self.D,
                            gamma = self.gamma)
                    
                    perturbation = perturbation + np.real(sim.curl_tilde_v(t,x,y))
                    
        return perturbation
    
    def curl_delta_v(self, t, x, y):
        '''
        Dimensional vorticity perturbation (curl of the velocity perturbation)
        in units of [1 / microsecond].
        
        Parameters
        ----------
        t : float [microsecond]
            Time coordinate.
        x : float or np.array of floats [nm]
            x coordinates(s). Must be the same shape as y.
            Must be the same length unit as the grain and interstitial widths.
        y : float or np.array of floats [nm]
            y coordinates(s). Must be the same shape as x.
            Must be the same length unit as the grain and interstitial widths.

        Returns
        -------
        float or np.array of floats with same shape as x and y
            Returns the value of the curl of the total velocity perturbation
            in units of [1 / microsecond] of the specified component at x and y, 
            or at all pairs of coordinates (x,y) if np.array of floats are given. 
        '''
        return self.a2 * self.curl_tilde_v(t, x, y)
    
    ##-----------------------------------------------------------------------
    ## Define kinetic, internal, and total energy - both normalized and dimensional
    
    def tilde_KE(self, t, x, y, lab_frame = False, perturb_from_mean = False):
        '''
        Normalized kinetic energy perturbation (per unit volume) 
        which subtracts out the baseline kinetic energy without perturbations.
        Normalized with the lab reference frame kinetic energy: 0.5 * rho * U**2
        lab_frame specifies whether to calculate in the lab reference frame,
        or the post-shock stationary reference frame
        
        Parameters
        ----------
        t : float [microsecond]
            Time coordinate.
        x : float or np.array of floats [nm]
            x coordinates(s). Must be the same shape as y.
            Must be the same length unit as the grain and interstitial widths.
        y : float or np.array of floats [nm]
            y coordinates(s). Must be the same shape as x.
            Must be the same length unit as the grain and interstitial widths.
        lab_frame: bool, default = False
            Specify if the kinetic energy perturbation should be calculated in
            postshock stationary reference frame (default of lab_frame = False),
            or in the lab refrence frame where the baseline x-velocity is U.
        perturb_from_mean : bool, default = False
            If perturb_from_mean is False (the default), the the density perturbation
            is calculated with respect to rho2_reference, from the provided reference 
            density. If perturn_from_mean is True, then the perturbation is calculated
            with respect to the mean density rho2.

        Returns
        -------
        float or np.array of floats with same shape as x and y
            Returns the value of the normalized kinetic energy perturbation 
            at x and y, or at all pairs of coordinates (x,y) if np.array of 
            floats are given. The energy is normalized with the lab reference 
            frame kinetic energy: 0.5 * rho * U**2, where rho is rho2 or
            rho2_reference, depending on the value of perturb_from_mean.

        '''
        if lab_frame:
            ke = (self.U + self.a2*self.tilde_vx(t,x,y))**2 
            ke = ke + (self.a2 * self.tilde_vy(t,x,y))**2
            ke = (1 + self.tilde_rho(t,x,y, perturb_from_mean)) * ke
            ke = ke / (self.U**2) - 1
        else:
            ke = self.tilde_vx(t,x,y)**2 + self.tilde_vy(t,x,y)**2
            ke = (1 + self.tilde_rho(t,x,y, perturb_from_mean)) * ke
            ke = ke * (self.a2 / self.U)**2
        
        return ke
    
    def delta_KE(self, t, x, y, lab_frame = False, perturb_from_mean = False):
        '''
        Kinetic energy perturbation per unit volume 
        in units of [g / (cm * microseconds^2)]
        
        Parameters
        ----------
        t : float [microsecond]
            Time coordinate.
        x : float or np.array of floats [nm]
            x coordinates(s). Must be the same shape as y.
            Must be the same length unit as the grain and interstitial widths.
        y : float or np.array of floats [nm]
            y coordinates(s). Must be the same shape as x.
            Must be the same length unit as the grain and interstitial widths.
        lab_frame: bool, default = False
            Specify if the kinetic energy perturbation should be calculated in
            postshock stationary reference frame (default of lab_frame = False),
            or in the lab refrence frame where the baseline x-velocity is U.
        perturb_from_mean : bool, default = False
            If perturb_from_mean is False (the default), the the density perturbation
            is calculated with respect to rho2_reference, from the provided reference 
            density. If perturn_from_mean is True, then the perturbation is calculated
            with respect to the mean density rho2.

        Returns
        -------
        float or np.array of floats with same shape as x and y
            Returns the value of the kinetic energy perturbation in units of 
            [g / (cm * microseconds^2)] at x and y, or at all pairs of 
            coordinates (x,y) if np.array of floats are given. The energy 
            perturbation is from 0 if lab_frame = False, and from 
            0.5 * rho * U**2 if lab_frame = True, where rho is rho2 or
            rho2_reference, depending on the value of perturb_from_mean.
        
        '''
        if perturb_from_mean:
            0.5 * self.rho2 * self.U**2 * self.tilde_KE(t,x,y, lab_frame = lab_frame, perturb_from_mean = True)
        else:
            return 0.5 * self.rho2_reference * self.U**2 * self.tilde_KE(t,x,y, lab_frame = lab_frame)
    
    def tilde_IE(self, t, x, y):
        '''
        Normalized internal energy perturbation (per unit volume) 
        which subtracts out the baseline internal energy without perturbations. 
        Normalized like pressure by gamma * p2.
        
        Parameters
        ----------
        t : float [microsecond]
            Time coordinate.
        x : float or np.array of floats [nm]
            x coordinates(s). Must be the same shape as y.
            Must be the same length unit as the grain and interstitial widths.
        y : float or np.array of floats [nm]
            y coordinates(s). Must be the same shape as x.
            Must be the same length unit as the grain and interstitial widths.

        Returns
        -------
        float or np.array of floats with same shape as x and y
            Returns the value of the normalized internal energy perturbation 
            at x and y, or at all pairs of coordinates (x,y) if np.array of 
            floats are given. The value is normalized like pressure by gamma * p2.
        '''
        
        return self.tilde_p(t,x,y) / (self.gamma - 1)
    
    def delta_IE(self, t, x, y):
        '''
        Internal energy perturbation per unit volume 
        in units of [g / (cm * microseconds^2)]
        
        Parameters
        ----------
        t : float [microsecond]
            Time coordinate.
        x : float or np.array of floats [nm]
            x coordinates(s). Must be the same shape as y.
            Must be the same length unit as the grain and interstitial widths.
        y : float or np.array of floats [nm]
            y coordinates(s). Must be the same shape as x.
            Must be the same length unit as the grain and interstitial widths.

        Returns
        -------
        float or np.array of floats with same shape as x and y
            Returns the value of the internal energy perturbation in units of 
            [g / (cm * microseconds^2)] at x and y, or at all pairs of coordinates 
            (x,y) if np.array of floats are given. 
        '''
        return self.gamma * self.p2 * self.tilde_IE(t, x, y)
    
    def integral_IE(self, t):
        '''
        Integral at time t of the normalized internal of energy perturbation 
        (per unit volume) integrated over the Lx x Ly box and taken per unit 
        area by dividing Lx*Ly. Normalized like pressure by gamma * p2.
        
        The only contributions to the internal energy integral are from the ky = 0
        Fourier modes, so we can analytically determine the integral for those modes
        and them sum.
        '''
        integral = 0
        
        #Add the perturbations from all mode combos other than the 0,0 constant mode
        for i in range(1, self.n_x_modes):
                    
            #Calculate normalized density perturbation amplitude
            epsilon_k = self.x_coeff[i] * self.y_coeff[0] / self.rho1
        
            sim = SingleMode.SingleModeSolution(
                    Nx = i, Ny = 0, M1 = self.M1, 
                    Lx = self.Lx, Ly = self.Ly,
                    epsilon_k = epsilon_k,
                    rho1 = self.rho1, 
                    D = self.D,
                    gamma = self.gamma)
            
            #Calculate the integral in x and t
            a = np.imag(sim.c)
            b = - (sim.omega + a * sim.M2*sim.a2) * t
            first_integral = (1/a) * (np.sin(a*sim.Lx + b) - np.sin(b))
            first_integral = np.real(sim.p0) * first_integral
            second_integral = (1/a) * (np.cos(a*sim.Lx + b) - np.cos(b))
            second_integral = np.imag(sim.p0) * second_integral
            #Add to the integral over the other modes
            integral = integral + (first_integral + second_integral)
            
        #Include the y-component Ly and divide the pressure integral by gamma-1 for IE
        integral = integral*self.Ly / (self.gamma-1)
        return integral / (self.Lx * self.Ly)
    
    ##-----------------------------------------------------------------------
    ## Define momentum - both normalized and dimensional
    
    def tilde_Mx(self, t, x, y, lab_frame = False, perturb_from_mean = False):
        '''
        Normalized momentum perturbation in x-direction (per unit volume) 
        which subtracts out the x-momentum without perturbations. 
        Normalized by rho * a2, using the post-shock sound speed, where rho 
        is rho2 or rho2_reference depending on the value of perturb_from_mean.
        
        Parameters
        ----------
        t : float [microsecond]
            Time coordinate.
        x : float or np.array of floats [nm]
            x coordinates(s). Must be the same shape as y.
            Must be the same length unit as the grain and interstitial widths.
        y : float or np.array of floats [nm]
            y coordinates(s). Must be the same shape as x.
            Must be the same length unit as the grain and interstitial widths.
        lab_frame: bool, default = False
            Specify if the momentum perturbation should be calculated in
            postshock stationary reference frame (default of lab_frame = False),
            or in the lab refrence frame where the baseline x-velocity is U.
        perturb_from_mean : bool, default = False
            If perturb_from_mean is False (the default), the the density perturbation
            is calculated with respect to rho2_reference, from the provided reference 
            density. If perturn_from_mean is True, then the perturbation is calculated
            with respect to the mean density rho2.

        Returns
        -------
        float or np.array of floats with same shape as x and y
            Returns the value of the normalized x-component of momentum perturbation
            (per unit volume) at x and y, or at all pairs of coordinates (x,y) 
            if np.array of floats are given. The momentum perturbation is from 0 
            if lab_frame = False, and from rho * U**2 if lab_frame = True, where 
            rho is rho2 or rho2_reference, depending on the value of perturb_from_mean.
            The value is normalized by rho * a2, using the post-shock sound speed,
            where rho is rho2 or rho2_reference depending on the value of perturb_from_mean.
        
        '''
        if lab_frame:
            tilde_rho = self.tilde_rho(t,x,y, perturb_from_mean)
            Mx = (1 + tilde_rho) * self.tilde_vx(t,x,y)
            Mx = Mx + tilde_rho  * self.U / self.a2 
        else:
            Mx = (1 + self.tilde_rho(t,x,y, perturb_from_mean)) * self.tilde_vx(t,x,y)
        
        return Mx
    
    def delta_Mx(self, t, x, y, lab_frame = False, perturb_from_mean = False):
        '''
        Momentum perturbation in the x-direction per unit volume 
        in units of [g / microseconds]
        
        Parameters
        ----------
        t : float [microsecond]
            Time coordinate.
        x : float or np.array of floats [nm]
            x coordinates(s). Must be the same shape as y.
            Must be the same length unit as the grain and interstitial widths.
        y : float or np.array of floats [nm]
            y coordinates(s). Must be the same shape as x.
            Must be the same length unit as the grain and interstitial widths.
        lab_frame: bool, default = False
            Specify if the momentum perturbation should be calculated in
            postshock stationary reference frame (default of lab_frame = False),
            or in the lab refrence frame where the baseline x-velocity is U.
        perturb_from_mean : bool, default = False
            If perturb_from_mean is False (the default), the the density perturbation
            is calculated with respect to rho2_reference, from the provided reference 
            density. If perturn_from_mean is True, then the perturbation is calculated
            with respect to the mean density rho2.

        Returns
        -------
        float or np.array of floats with same shape as x and y
            Returns the value of the x-component of momentum perturbation in units
            of [g / microseconds] at x and y, or at all pairs of coordinates (x,y) 
            if np.array of floats are given. The momentum perturbation is from 0 if
            lab_frame = False, and from rho * U**2 if lab_frame = True, 
            where rho is rho2 or rho2_reference, depending on the value of 
            perturb_from_mean.
        
        '''
        if perturb_from_mean:
            return self.rho2 * self.a2 * self.tilde_Mx(t,x,y, lab_frame, perturb_from_mean=True)
        else:
            return self.rho2_reference * self.a2 * self.tilde_Mx(t,x,y, lab_frame)
    
    def tilde_My(self, t, x, y, perturb_from_mean = False):
        '''
        Normalized momentum perturbation in y-direction (per unit volume) 
        which subtracts out the y-momentum without perturbations. 
        Normalized by rho * a2, using the post-shock sound speed, where rho 
        is rho2 or rho2_reference depending on the value of perturb_from_mean.
        
        Parameters
        ----------
        t : float [microsecond]
            Time coordinate.
        x : float or np.array of floats [nm]
            x coordinates(s). Must be the same shape as y.
            Must be the same length unit as the grain and interstitial widths.
        y : float or np.array of floats [nm]
            y coordinates(s). Must be the same shape as x.
            Must be the same length unit as the grain and interstitial widths.
        perturb_from_mean : bool, default = False
            If perturb_from_mean is False (the default), the the density perturbation
            is calculated with respect to rho2_reference, from the provided reference 
            density. If perturn_from_mean is True, then the perturbation is calculated
            with respect to the mean density rho2.

        Returns
        -------
        float or np.array of floats with same shape as x and y
            Returns the value of the normalized y-component of momentum perturbation
            (per unit volume) at x and y, or at all pairs of coordinates (x,y) 
            if np.array of floats are given. The momentum perturbation is from 0.
            The value is normalized by rho * a2, using the post-shock sound speed,
            where rho is rho2 or rho2_reference depending on the value of 
            perturb_from_mean.

        '''
        
        My = (1 + self.tilde_rho(t,x,y, perturb_from_mean)) * self.tilde_vy(t,x,y)
        
        return My
    
    def delta_My(self, t, x, y, perturb_from_mean = False):
        '''
        Momentum perturbation in the y-direction per unit volume 
        in units of [g / microseconds]
        
        Parameters
        ----------
        t : float [microsecond]
            Time coordinate.
        x : float or np.array of floats [nm]
            x coordinates(s). Must be the same shape as y.
            Must be the same length unit as the grain and interstitial widths.
        y : float or np.array of floats [nm]
            y coordinates(s). Must be the same shape as x.
            Must be the same length unit as the grain and interstitial widths.
        perturb_from_mean : bool, default = False
            If perturb_from_mean is False (the default), the the density perturbation
            is calculated with respect to rho2_reference, from the provided reference 
            density. If perturn_from_mean is True, then the perturbation is calculated
            with respect to the mean density rho2.

        Returns
        -------
        float or np.array of floats with same shape as x and y
            Returns the value of the y-component of momentum perturbation in units
            of [g / microseconds] at x and y, or at all pairs of coordinates (x,y) 
            if np.array of floats are given. The momentum perturbation is from 0.
        '''
        if perturb_from_mean:
            return self.rho2 * self.a2 * self.tilde_My(t,x,y, perturb_from_mean=True)
        else:
            return self.rho2_reference * self.a2 * self.tilde_My(t,x,y)
    
    def tilde_abs_M(self, t, x, y, lab_frame = False, perturb_from_mean = False):
        '''
        Normalized magnitude of total momentum perturbation (per unit volume) 
        which subtracts out the momentum without perturbations. 
        Normalized by rho * a2, using the post-shock sound speed, where rho 
        is rho2 or rho2_reference depending on the value of perturb_from_mean.
        
        Parameters
        ----------
        t : float [microsecond]
            Time coordinate.
        x : float or np.array of floats [nm]
            x coordinates(s). Must be the same shape as y.
            Must be the same length unit as the grain and interstitial widths.
        y : float or np.array of floats [nm]
            y coordinates(s). Must be the same shape as x.
            Must be the same length unit as the grain and interstitial widths.
        lab_frame: bool, default = False
            Specify if the momentum perturbation should be calculated in
            postshock stationary reference frame (default of lab_frame = False),
            or in the lab refrence frame where the baseline x-velocity is U.
        perturb_from_mean : bool, default = False
            If perturb_from_mean is False (the default), the the density perturbation
            is calculated with respect to rho2_reference, from the provided reference 
            density. If perturn_from_mean is True, then the perturbation is calculated
            with respect to the mean density rho2.

        Returns
        -------
        float or np.array of floats with same shape as x and y
            Returns the value of the normalized total momentum perturbation
            (per unit volume) at x and y, or at all pairs of coordinates (x,y) 
            if np.array of floats are given. The value is normalized by rho * a2, 
            using the post-shock sound speed, where rho is rho2 or rho2_reference 
            depending on the value of perturb_from_mean.
        '''
        M = self.tilde_Mx(t,x,y, lab_frame, perturb_from_mean)**2
        return np.sqrt(M + self.tilde_My(t,x,y, perturb_from_mean)**2)