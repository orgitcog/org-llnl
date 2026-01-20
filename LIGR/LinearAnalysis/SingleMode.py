#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Grace Li (li85)
Date: 8/4/2021

SingleModeSolution class to represent analytic solution of a shock through 
a single Fourier mode density perturbation from the the equations in the 
Velikovich et al. 2007 paper: 
''Shock front distortion and Richtmyer-Meshkov-type growth caused by 
a small preshock nonuniformity''
Phys. Plasmas 14, 072706 (2007)

"""
import warnings
import numpy as np
from numpy import pi as pi
import scipy.integrate as integrate
import math

# import matplotlib.pyplot as plt

class SingleModeSolution:
    
    #Reference frame with stationary post-shock
    vx2 = 0
    
    def __init__(self, Nx, Ny, M1, 
                 Lx = 1.0, Ly = 1.0,
                 epsilon_k = 1.0,
                 rho1 = 1.0, 
                 D = 2.39,
                 gamma = 5/3):
        '''
        Constructs the necessary attributes for the SingleModeSolution object.
        
        Parameters
        ----------
        Nx : int [dimensionless]
            Integer portion of the wavenumber in the x-direction. 
            The wavenumber is kx = 2*pi*Nx/Lx = 2*pi / lambda_x.
            The wavelength is lambda_x = Lx / Nx.
        Ny : int [dimensionless]
            Integer portion of the wavenumber in the y-direction. 
            The wavenumber is ky = 2*pi*Ny/Ly = 2*pi / lambda_y.
            The wavelength is lambda_y = Ly / Ny.
        M1 : float [dimensionless]
            Mach number of the incident shock.
            M1 = D/a1 where D is the shock speed if the preshock fluid is at rest,
            and a1 is the speed of sound in the preshock fluid.
        Lx : float [cm], default = 1.0
            Length of one period in the x-direction in cm.
        Ly : float [cm], default = 1.0
            Length of one period in the y-direction in cm.
        epsilon_k [dimensionless]: float, default = 1.0
            Normalized amplitude of the density perturbations
            (i.e. the amplitude of delta_rho / rho1).
            In theory the linear analysis should hold for epsilon_k << 1.
            Mostly, epsilon_k scales the resulting perturbations values as well
            as the shock front amplitude.
        rho1 : float [g/cm3], default = 1.0
            The bulk density of the preshock fluid in g/cm3. 
            Density perturbations are taken to be perturbations from rho1. 
        D : float [cm / microsecond], default = 2.39
            Shock speed (relative to the lab reference frame) in cm / microsecond,
            which is also 10 * km/second
        gamma : float [dimensionless], default = 5/3
            Adiabatic index (also known as heat capacity ratio).
            This analysis uses gamma in the ideal gas equation of state.

        Returns
        -------
        A fully constructed instance of the SingleModeSolution object.

        '''
        
        # Initialize parameters
        self.Nx = Nx #Integer for x-direction wave
        self.Lx = Lx #Period in x-direction
        self.kx = 2*pi*Nx/self.Lx #Wave number in x
        
        self.Ny = Ny #Integer for y-direction wave
        self.Ly = Ly #Period in y-direction
        self.ky = 2*pi*Ny/self.Ly #Wave number in y
        
        self.D = D #Shock speed relative to lab reference frame
        self.M1 = M1 #Pre-shock Mach number
        self.epsilon_k = epsilon_k #Density perturbation magnitude
        self.rho1 = rho1 #Preshock bulk density
        self.gamma = gamma #Adiabatic gamma for ideal gas EOS
        
        #Calculate the preshock speed of sound
        self.a1 = D/self.M1
        
        #Calculate the preshock pressure form an ideal gas EOS
        #The units are [g / (cm * microseconds^2)]
        self.p1 = self.rho1 * (self.a1)**2 / self.gamma
        
        # Calculate average post shock quantities
        self.rho2 = ( (gamma+1)*(M1**2) / ((gamma-1)*(M1**2) + 2) )* self.rho1
        self.p2 = ((2*gamma*(M1**2) - gamma + 1) / (gamma + 1) )* self.p1
        a2 = math.sqrt( (2*gamma*(M1**2) - gamma + 1) * ((gamma-1)*(M1**2) + 2) )
        self.a2 = ( a2 / ( (gamma+1)*M1 ) )* self.a1
        M2 = ( (gamma-1)*(M1**2) + 2 ) / (2*gamma*(M1**2) - gamma + 1)
        self.M2 = math.sqrt(M2)
    
        #Calculate the compression ratio rho2/rho1
        self.R = self.rho2 / self.rho1
        
        #Calculate the post shock fluid speed relative to the preshock fluid
        #To be used as the lab reference frame velocity for normalization
        #U = Difference in velocity of pre and post shock fluids
        self.U = self.D - self.M2*self.a2

        # #Pre-shock fluid speed if in reference frame with stationary post-shock fluid
        # self.vx1 = -self.U
        
        # Determine if we satisfy long or short wavelength condition
        if self.Ny != 0 and (self.kx / self.ky) < ( math.sqrt(1-self.M2**2) / (self.R*self.M2) ):
            self.longwavelength = True
        else:
            self.longwavelength = False
        
        # Calculate the frequency omega
        self.omega = self.kx * self.D
        
        #For the ky = 0, calculate constant c, representing ky*eta in the sonic exponent     
        if Ny == 0:
            self.c = complex(0, -self.omega / (self.a2 * (self.M2 + 1)))
        #Otherwise, calculate zeta0 and eta as outlined in the paper
        else:
            #calculate zeta0 from Eqn 25 in the paper
            zeta0 = (self.kx * self.R * self.M2) 
            self.zeta0 = zeta0 / (self.ky * math.sqrt(1 - self.M2**2))
            
            # Calculate eta (a complex number) from Eqns 27 and 28
            if self.longwavelength:
                eta = complex(math.sqrt(1-self.zeta0**2), self.M2*self.zeta0)
                self.eta = eta / (math.sqrt(1-self.M2**2))
            else:
                eta = complex(0, -math.sqrt(self.zeta0**2 - 1) + self.M2*self.zeta0 )
                self.eta = eta / (math.sqrt(1-self.M2**2))
    
            
        #Calculate amplitudes
        self.get_amplitudes()
        
        #Vectorize functions to accept numpy input for plotting
        # self.vectorize_functions()
        
    ##-----------------------------------------------------------------------
    #Function for the shock amplitude delta_xs
    def delta_xs(self, t, y = None):
        '''
        Calculate the shock front perturbation delta_xs (in cm)
        from Eqns 32, 35, and 36 in the paper. 

        Parameters
        ----------
        t : float [microsecond]
            Time.
        y : float [cm], optional
            y-coordinate. If no y value is provided, then the function returns
            the shock amplitude (complex with both real and imaginary components)
            without calculating the value in terms of y-coordinate position.
        
        Returns
        -------
        float [cm]
            The shock front perturbation at (t,y). If no y value is provided, then
            the shock front oscillation amplitude (full complex value) is returned.

        '''
        Q = self.R*(self.M2**2) / math.sqrt(1-self.M2**2)
        
        #Eqn 32 in the paper
        if self.longwavelength:
            beta1 = 1 - ( 1 / ((self.M1**4)*(self.M2**2)) )
            beta2 = 2*(2*self.M2**2 - 1 - 1/(self.M1**2))
            beta3 = (1 + 1/(self.M1**2))**2 - 4*(self.M2**2)
            
            phi1 = 2*self.M2 * (self.M1**2) * self.zeta0 * math.sqrt(1-self.zeta0**2)
            phi1 = phi1 / (self.M1**2 - (self.M1**2 + 1)*(self.zeta0**2))
            phi1 = math.atan(phi1) 
            if phi1 < 0:
                phi1 = phi1 + pi
                
            phi2 = self.zeta0 / ( (self.M1**2) * self.M2 * math.sqrt(1-self.zeta0**2) )
            phi2 = math.atan(phi2)
            if phi2 < 0:
                phi2 = phi2 + pi
                
            self.phi = phi1 - phi2
            # if phi < 0:
            #     phi = phi + math.pi
            
            coeff = 1 - beta1*(self.zeta0**2)
            coeff = coeff / (1 + beta2*(self.zeta0**2) + beta3*(self.zeta0**4))
            coeff = - self.epsilon_k * Q * math.sqrt(coeff)
            
            exponential = np.exp( (self.phi - self.omega*t) * 1j)
            xt_value = coeff * exponential / self.ky

        #Special shortwavelength case for ky = 0 from Eqn 36 of the paper
        elif self.Ny == 0:
            top = (self.M1**2) * self.M2 + 1
            bottom = 2*(self.M1**2) * self.M2 + (self.M1**2) + 1
            
            coeff = - self.epsilon_k * top / bottom
            
            exponential = np.exp( (pi/2 - self.omega*t ) * 1j)
            xt_value = coeff * exponential / self.kx

        #ky != 0, short-wavelength form in Eqn 35 in the paper
        else:
            top = self.zeta0 + (self.M1**2) * self.M2 * math.sqrt(self.zeta0**2 - 1)
            bottom = (self.M1**2 + 1)*(self.zeta0**2) - self.M1**2
            bottom = bottom + 2*(self.M1**2) * self.M2 * self.zeta0 * math.sqrt(self.zeta0**2-1)
            bottom = self.M2 * bottom
            
            coeff = - self.epsilon_k * Q * top / bottom
            
            exponential = np.exp( (pi/2 - self.omega*t ) * 1j)
            xt_value = coeff * exponential / self.ky
                        
        #If y is None, just return the x,t component value
        if y is None:
            return xt_value

        #Include the y-component (this is 1 if ky = 0)
        return xt_value * np.cos(self.ky * y)
        

    def delta_rho_rho1(self, t, x, y = None):
        '''
        Dimensionless density perturbation (normalized with rho1)
        delta rho_k / rho1 from Equation 20 in the paper
        '''
        exponent = self.kx * (x + self.U * t)
        xt_value = self.epsilon_k * np.exp(-exponent * 1j)
        
        #If y is None, just return the x,t component value
        if y is None:
            return xt_value
        
        #Include the y-component
        return xt_value * np.cos(self.ky * y)
        
    ##-----------------------------------------------------------------------
    def get_amplitudes(self):
        '''
        Calculate the amplitudes of perturbations from the Rankine-Hugoniot
        Eqns 16-19 and the shock amplitude delta_xs
        '''
        #Pressure perturbation amplitude
        #From manipulating Eqn 19 in the paper
        coeff1 = 2*(self.M2**2)*self.R / (self.gamma + 1)
        coeff2 = -4 * self.M2 * self.omega / (self.a2 * (self.gamma + 1))
        coeff2 = complex(0, coeff2)
        self.p0 = coeff1 * self.epsilon_k + coeff2 * self.delta_xs(0)
        
        #Sonic wave amplitude of x-component of velocity
        #If ky = 0, calculate using our definition of c
        if self.Ny == 0:
            coeff = complex(self.c * self.M2 * self.a2, self.omega)
            coeff = self.c * self.a2 / coeff
            self.vx0_s = coeff * self.p0
        #Otherwise, calculate from Eqn 29 in the paper
        else:
            coeff = complex(self.M2 * self.eta, self.zeta0 * math.sqrt(1-self.M2**2))
            coeff = self.eta / coeff
            self.vx0_s = coeff * self.p0
        
        #Sonic wave amplitude of y-component of velocity
        #If ky = 0, there is no y-velocity perturbation
        if self.Ny == 0:
            self.vy0_s = 0 
        #Otherwise, calculate from Eqn 29 in the paper
        else:
            coeff = complex(self.M2 * self.eta, self.zeta0 * math.sqrt(1-self.M2**2))
            coeff = -1 / coeff
            self.vy0_s = coeff * self.p0
        
        #Vortex wave amplitude of x-component of velocity
        #If ky = 0, there is no vortex perturbation
        if self.Ny == 0:
            self.vx0_v = 0
            self.vy0_v = 0
        #Otherwise, calculate x and y velocity vortex amplitudes
        else:
            #From manipulating Eqn 17 in the paper
            coeff1 = - self.M2 * (self.R-1) / 2
            coeff2 = (self.M1**2 + 1)/(2*(self.M1**2)*self.M2)
            self.vx0_v = coeff1 * self.epsilon_k + coeff2 * self.p0 - self.vx0_s
        
            #Vortex wave amplitude of y-component of velocity
            #From Eqn 31 in the paper
            coeff = complex(0, self.zeta0 * math.sqrt(1-self.M2**2) / self.M2)
            self.vy0_v = coeff * self.vx0_v
        
        #Sonic wave amplitude of density
        #From Eqn 6 of the paper
        if self.Ny == 0:
            coeff = complex(self.c * self.M2 * self.a2, self.omega)
            coeff = self.c * self.a2 / coeff
            self.rho0_s = coeff * self.vx0_s
        else:
            coeff = complex(self.ky*self.eta * self.M2*self.a2, self.omega)
            coeff = self.ky * self.a2 / coeff
            self.rho0_s = coeff * ( self.eta * self.vx0_s + self.vy0_s )
        
        #Entropy wave of density
        #This is a constant as seen from Eqn 6
        #From Eqn 18 of the paper
        self.rho0_e = -self.rho0_s + self.p0 /((self.M1**2)*(self.M2**2)) + self.epsilon_k
        
    ##-----------------------------------------------------------------------
    
    ## Define sonic and vortex/entropy form exponents for re-use in pertrubations
    def sonic_exp(self, t, x):
        exponent = -1j * self.omega * t
        #If ky = 0, calculate the sonic exponent with c
        if self.Ny == 0:
            exponent = exponent + self.c * (x - self.M2*self.a2 * t)
        #Otherwise calculate the sonic exponent given in Eqn 22
        else:
            exponent = exponent + self.ky * self.eta * (x - self.M2*self.a2 * t)
        return np.exp(exponent)
    
    def vortex_exp(self, x):
        #No vortex exponent in ky = 0 case
        if self.Ny == 0:
            return 0
        
        exponent = -1j * self.omega * x / (self.M2 *self.a2) 
        return np.exp(exponent)

    def entropy_exp(self, x):
         exponent = -1j * self.omega * x / (self.M2 *self.a2) 
         return np.exp(exponent)
      
    ## Define sonic and vortex/entropy x,t components taking into account shock front
    ## Return 0 (default no perturbation) if pre shock, and sonic/vortex/entropy
    ## exponent if post shock. Handles the scalar vs np.array input case
    def exp_with_shock_front(self, component, t, x, y):
        
        #Calculate the perturbation only if we're post shock
        #The default perturbation value is 0 if we're pre shock
        
        #Determine which if a scalar x is post shock, or which indicies 
        #if x is a np.array are post shock.
        #Print an informative message to check dimensions if they are wrong
        try: 
            post_shock = (x <= self.M2*self.a2 * t + np.real(self.delta_xs(t,y)))
        except:
            print("Check input dimensions. t should be a scalar representing time.")
            print("""x and y should either both be scalars, or both be numpy arrays
                  of the same shape, in which case the function will be evaluated at
                  all pairs (x,y) from the arrays and return output of the same shape.""")
            raise
        
        # Handle the scalar x case
        if np.isscalar(x):
            xt_value = 0
            #Calculate the post-shock perturbation if x is post-shock
            if post_shock:
                if component in ["sonic", "s"]:
                    xt_value = self.sonic_exp(t,x)
                elif component in ["vortex", "v"]:
                    xt_value = self.vortex_exp(x)
                elif component in ["entropy", "e"]:
                    xt_value = self.entropy_exp(x)
            
        # Handle the np.array x case
        else:
            xt_value = np.zeros(x.shape, dtype=complex)
            if component in ["sonic", "s"]:
                    xt_value[post_shock] = self.sonic_exp(t, x[post_shock])
            elif component in ["vortex", "v"]:
                xt_value[post_shock] = self.vortex_exp(x[post_shock])
            elif component in ["entropy", "e"]:
                xt_value[post_shock] = self.entropy_exp(x[post_shock])
            
        return xt_value

    ##-----------------------------------------------------------------------
    ## Define normalized (tilde) post-shock perturbations
        
    def tilde_p(self, t, x, y = None):
        '''
        Pressure dimensionless perturbation (normalized with rho2)
        This has sonic component only.
        '''        
        #If no y value is provided, return the post-shock perturbation in x,t only
        if y is None:
            return self.p0 * self.sonic_exp(t,x)
        #Otherwise, include the y-component (this is 1 if ky = 0)
        return self.p0 * self.exp_with_shock_front("sonic", t, x, y) * np.cos(self.ky * y)
    
    
    def tilde_vx_s(self, t, x, y = None):
        '''
        Sonic dimensionless velocity perturbation in x-component (normalized with a2)
        '''  
        #If no y value is provided, return the post-shock perturbation in x,t only
        if y is None:
            return self.vx0_s * self.sonic_exp(t,x)
        #Otherwise, include the y-component (this is 1 if ky = 0)
        return self.vx0_s * self.exp_with_shock_front("sonic", t, x, y) * np.cos(self.ky * y)
        
        
    def tilde_vy_s(self, t, x, y = None):
        '''
        Sonic dimensionless velocity perturbation in y-component (normalized with a2)
        '''  
        #If no y value is provided, return the post-shock perturbation in x,t only
        if y is None:
            return self.vy0_s * self.sonic_exp(t,x)
        #Otherwise, include the y-component 
        #(if ky = 0, this is 0, and there is no y-velocity perturbation)
        return self.vy0_s * self.exp_with_shock_front("sonic", t, x, y) * np.sin(self.ky * y)
    
    def tilde_vx_v(self, t, x, y = None):
        '''
        Vortex dimensionless velocity perturbation in x-component (normalized with a2)
        ''' 
        #If no y value is provided, return the post-shock perturbation in x,t only
        if y is None:
            return self.vx0_v * self.vortex_exp(x)
        #Otherwise, include the y-component (this is 1 if ky = 0)
        return self.vx0_v * self.exp_with_shock_front("vortex",t, x, y) * np.cos(self.ky * y)
        
    def tilde_vy_v(self, t, x, y = None):
        '''
        Vortex dimensionless velocity perturbation in y-component (normalized with a2)
        '''
        #If no y value is provided, return the post-shock perturbation in x,t only
        if y is None:
            return self.vy0_v * self.vortex_exp(x)
        #Otherwise, include the y-component 
        #(if ky = 0, this is 0, and there is no y-velocity perturbation)
        return self.vy0_v * self.exp_with_shock_front("vortex",t, x, y) * np.sin(self.ky * y)
    
    def tilde_vx(self, t, x, y = None):
        '''
        Combined dimensionless velocity perturbation in x-component (normalized with a2)
        '''
        return self.tilde_vx_s(t,x,y) + self.tilde_vx_v(t,x,y)
    
    def tilde_vy(self, t, x, y = None):
        '''
        Combined dimensionless velocity perturbation in y-component (normalized with a2)
        '''
        return self.tilde_vy_s(t,x,y) + self.tilde_vy_v(t,x,y)

    def tilde_rho_s(self, t, x, y = None):
        '''
        Sonic dimensionless density perturbation (normalized with rho2)
        '''
        #If no y value is provided, return the post-shock perturbation in x,t only
        if y is None:
            return self.rho0_s * self.sonic_exp(t,x)
        #Otherwise, include the y-component (this is 1 if ky = 0)
        return self.rho0_s * self.exp_with_shock_front("sonic", t, x, y) * np.cos(self.ky * y)
        
    
    def tilde_rho_e(self, t, x, y = None):
        '''
        Entropy dimensionless density perturbation (normalized with rho2)
        '''
        #If no y value is provided, return the post-shock perturbation in x,t only
        if y is None:
            return self.rho0_e * self.entropy_exp(x)
        #Otherwise, include the y-component (this is 1 if ky = 0)
        return self.rho0_e * self.exp_with_shock_front("entropy",t, x, y) * np.cos(self.ky * y)

    
    def tilde_rho(self, t, x, y = None):
        '''
        Combined dimensionless density perturbation (normalized with rho2)
        '''
        #Determine if a scalar x is pre/post shock, or which indicies 
        #if x is a np.array are pre/post shock.
        #Print an informative message to check dimensions if they are wrong
        try: 
            post_shock = (x <= self.M2*self.a2 * t + np.real(self.delta_xs(t,y)))
            pre_shock = np.invert(post_shock)
        except:
            print("Check input dimensions. t should be a scalar representing time.")
            print("""x and y should either both be scalars, or both be numpy arrays
                  of the same shape, in which case the function will be evaluated at
                  all pairs (x,y) from the arrays and return output of the same shape.""")
            raise
            
        # Handle the scalar x case
        if np.isscalar(x):
            if post_shock:
                return self.tilde_rho_s(t,x,y) + self.tilde_rho_e(t,x,y)
            else:
                return (self.rho1 / self.rho2) * self.delta_rho_rho1(t,x,y)
            
        # Handle the np.array x case
        else:
            z = np.zeros(x.shape, dtype=complex)
            z[post_shock] = (self.tilde_rho_s(t,x[post_shock],y[post_shock])
                             + self.tilde_rho_e(t,x[post_shock],y[post_shock]))
            z[pre_shock] = (self.rho1/self.rho2) * self.delta_rho_rho1(t,x[pre_shock],y[pre_shock])
            return z   
        
    ##-----------------------------------------------------------------------
    ## Get the dimensional (delta) version of perturbations.
    ## These quantities must be real, physical values
    ## Relations between dimensional (delta) and nondimensional (tilde)
    ## values comes from Eqn 5 in the paper
    
    def delta_rho_s(self, t, x, y = None):
        '''
        Sonic real and dimensional density perturbation
        in units of [g / cm3]
        '''
        perturbation = self.rho2 * self.tilde_rho_s(t,x,y)
        return np.real(perturbation)
    
    def delta_rho_e(self, t, x, y = None):
        '''
        Entropy real and dimensional density perturbation
        in units of [g / cm3]
        '''
        perturbation = self.rho2 * self.tilde_rho_e(t,x,y)
        return np.real(perturbation)
    
    def delta_rho(self, t, x, y = None):
        '''
        Combined real and dimensional density perturbation
        in units of [g / cm3]
        '''
        perturbation = self.rho2 * self.tilde_rho(t,x,y)
        return np.real(perturbation)
    
    def delta_p(self, t, x, y = None):
        '''
        Total (sonic only) real and dimensional pressure perturbation
        in units of [g / (cm * microseconds^2)]
        '''
        perturbation = self.gamma * self.p2 * self.tilde_p(t,x,y)
        return np.real(perturbation)
     
    def delta_vx_s(self, t, x, y = None):
        '''
        Sonic real and dimensional velocity perturbation in x-component
        in units of [cm / microsecond]
        '''
        perturbation = self.a2 * self.tilde_vx_s(t,x,y)
        return np.real(perturbation)
    
    def delta_vx_v(self, t, x, y = None):
        '''
        Vortex real and dimensional velocity perturbation in x-component
        in units of [cm / microsecond]
        '''
        perturbation = self.a2 * self.tilde_vx_v(t,x,y)
        return np.real(perturbation)
    
    def delta_vx(self, t, x, y = None):
        '''
        Combined real and dimensional velocity perturbation in x-component
        in units of [cm / microsecond]
        '''
        return self.delta_vx_s(t,x,y) + self.delta_vx_v(t,x,y)
    
    def delta_vy_s(self, t, x, y = None):
        '''
        Sonic real and dimensional velocity perturbation in y-component
        in units of [cm / microsecond]
        '''
        perturbation = self.a2 * self.tilde_vy_s(t,x,y)
        return np.real(perturbation)
    
    def delta_vy_v(self, t, x, y = None):
        '''
        Vortex real and dimensional velocity perturbation in y-component
        in units of [cm / microsecond]
        '''
        perturbation = self.a2 * self.tilde_vy_v(t,x,y)
        return np.real(perturbation)
    
    def delta_vy(self, t, x, y = None):
        '''
        Combined real and dimensional velocity perturbation in y-component
        in units of [cm / microsecond]
        '''
        return self.delta_vy_s(t,x,y) + self.delta_vy_v(t,x,y)
    
    ##-----------------------------------------------------------------------
    ## Calculate the divergence and curl of the perturbed velocity
    
    def div_tilde_v(self, t, x, y, component='total'):
        '''
        Normalized divergence of the perturbed velocity. Normalized by a2.
        By default, the total divergence is returned. If component = 'sonic' 
        or 'vortex', then only the divergence of the sonic or vortex 
        velocity perturbation respectively will be returned.
        '''
        
        #Determine which if a scalar x is post shock, or which indicies 
        #if x is a np.array are post shock.
        #Print an informative message to check dimensions if they are wrong
        try: 
            post_shock = (x <= self.M2*self.a2 * t + np.real(self.delta_xs(t,y)))
        except:
            print("Check input dimensions. t should be a scalar representing time.")
            print("""x and y should either both be scalars, or both be numpy arrays
                  of the same shape, in which case the function will be evaluated at
                  all pairs (x,y) from the arrays and return output of the same shape.""")
            raise
        
        #Handle the scalar case separately
        if np.isscalar(x):
            
            #If we're preshock, return 0 since there's no velocity perturbation
            if not post_shock:
                return 0
            
            #Calculate sonic component of the divergence of the perturbed velocity
            if self.Ny == 0:
                div_s = np.real(self.c * self.tilde_vx_s(t,x))
            else:
                div_s = np.real(self.eta * self.tilde_vx_s(t,x))
                div_s = div_s + np.real(self.tilde_vy_s(t,x))
                div_s = self.ky * np.cos(self.ky * y) * div_s
            if component in ['sonic']:
                return div_s 
        
            #Calculate vortex component of the divergence of the perturbed velocity.
            if self.Ny == 0:
                div_v = 0 #no vortex velocity component for ky=0 case
            else:
                div_v = np.real( -1j * self.omega/(self.M2*self.a2) * self.tilde_vx_v(t,x))
                div_v = div_v + self.ky * np.real(self.tilde_vy_v(t,x))
                div_v = np.cos(self.ky * y) * div_v
            if component in ['vortex']:
                return div_v
            
            return div_s + div_v
            
        #Handle the case were x and y are numpy arrays
        #The default perterbed velocity and therefore divergence is 0 if we are pre-shock
        div_s = np.zeros(x.shape)
        div_v = np.zeros(x.shape)
        
        x_postshock = x[post_shock] #post_shock x values
        y_postshock = y[post_shock] #post shock y values
        
        #Calculate sonic component of the divergence of the perturbed velocity.
        if self.Ny == 0:
            div_s[post_shock] = np.real(self.c * self.tilde_vx_s(t, x_postshock))
        else:
            z = np.real(self.eta * self.tilde_vx_s(t, x_postshock))
            z = z + np.real(self.tilde_vy_s(t, x_postshock))
            z = self.ky * np.cos(self.ky * y_postshock) * z
            div_s[post_shock] = z
        if component in ['sonic']:
            return div_s
        
        #Vortex component of the divergence of the perturbed velocity.
        #no vortex velocity component for ky=0 case
        if self.Ny != 0:
            div = np.real( -1j * self.omega/(self.M2*self.a2) * self.tilde_vx_v(t, x_postshock))
            div = div + self.ky * np.real(self.tilde_vy_v(t, x_postshock))
            div = np.cos(self.ky * y_postshock) * div
            div_v[post_shock] = div
        if component in ['vortex']:
            return div_v
        
        return div_s + div_v
    
    def curl_tilde_v(self, t, x, y, component='total'):
        '''
        Normalized perturbed vorticity (curl of the perturbed velocity).
        Normalized by a2.
        By default, the total curl is returned. If component = 'sonic' 
        or 'vortex', then only the curl of the sonic or vortex 
        velocity perturbation respectively will be returned.
        '''
        
        #Determine which if a scalar x is post shock, or which indicies 
        #if x is a np.array are post shock.
        #Print an informative message to check dimensions if they are wrong
        try: 
            post_shock = (x <= self.M2*self.a2 * t + np.real(self.delta_xs(t,y)))
        except:
            print("Check input dimensions. t should be a scalar representing time.")
            print("""x and y should either both be scalars, or both be numpy arrays
                  of the same shape, in which case the function will be evaluated at
                  all pairs (x,y) from the arrays and return output of the same shape.""")
            raise
            
        #Handle the scalar case separately
        if np.isscalar(x):
            
            #If we're preshock, return 0 since there's no velocity perturbation
            if not post_shock:
                return 0
            
            #There is no curl for the ky = 0 case, return 0 
            if self.Ny == 0:
                return 0
            
            #Sonic component of the vorticity (curl of the perturbed velocity)
            curl_s = np.real(self.eta * self.tilde_vy_s(t,x))
            curl_s = curl_s + np.real(self.tilde_vx_s(t,x))
            curl_s = self.ky * np.sin(self.ky * y) * curl_s
            if component in ['sonic']:
                return curl_s
            
            #Vortex component of the vorticity (curl of the perturbed velocity)
            curl_v = np.real(1j * self.omega/(self.M2*self.a2) * self.tilde_vy_v(t,x))
            curl_v = curl_v + self.ky * np.real(self.tilde_vx_v(t,x))
            curl_v = np.sin(self.ky * y) * curl_v
            if component in ['vortex']:
                return curl_v
            
            return curl_s + curl_v
        
        #Handle the case were x and y are numpy arrays
        #The default perterbed velocity and therefore divergence is 0 if we are pre-shock
        curl_s = np.zeros(x.shape)
        curl_v = np.zeros(x.shape)
        
        x_postshock = x[post_shock] #post_shock x values
        y_postshock = y[post_shock] #post shock y values
        
        #There is no curl for the ky = 0 case, return 0 
        if self.Ny == 0:
            return np.zeros(x.shape)
        
        #Sonic component of the vorticity (curl of the perturbed velocity)
        curl = np.real(self.eta * self.tilde_vy_s(t, x_postshock))
        curl = curl + np.real(self.tilde_vx_s(t, x_postshock))
        curl = self.ky * np.sin(self.ky * y_postshock) * curl
        curl_s[post_shock] = curl
        if component in ['sonic']:
            return curl_s
        
        #Vortex component of the vorticity (curl of the perturbed velocity)
        curl = np.real(1j * self.omega/(self.M2*self.a2) * self.tilde_vy_v(t, x_postshock))
        curl = curl + self.ky * np.real(self.tilde_vx_v(t, x_postshock))
        curl = np.sin(self.ky * y_postshock) * curl
        curl_v[post_shock] = curl
        if component in ['vortex']:
            return curl_v
        
        return curl_s + curl_v
    
    ##-----------------------------------------------------------------------
    ## Calculate the kinetic and total energy
    
    def tilde_KE(self, t, x, y):
        '''
        Normalized kinetic energy perturbation (per unit volume) 
        which subtracts out the baseline kinetic energy without perturbations. 
        Normalized with the lab reference frame kinetic energy: 0.5 * rho2 * U**2
        '''
        
        ke = np.real(self.tilde_vx(t,x,y))**2 + np.real(self.tilde_vy(t,x,y))**2
        ke = (1 + np.real(self.tilde_rho(t,x,y))) * ke
        
        ke = ke * (self.a2 / self.U)**2
        return ke
    
    def delta_KE(self, t, x, y):
        '''
        Kinetic energy perturbation per unit volume 
        in units of [g / (cm * microseconds^2)]
        '''
        return 0.5 * self.rho2 * self.U**2 * self.tilde_KE(t,x,y)
    
    
    def tilde_TE(self, t, x, y):
        '''
        Normalized total energy perturbation for an ideal gas (per unit volume) 
        which subtracts out the baseline total energy without perturbations.
        Normalized with the lab reference frame total energy: 
            p2 / (gamma - 1) + 0.5 * rho2 * U**2
        '''
        #Calculate unnormalized perturbed kinetic energy
        ke = self.delta_KE(t, x, y)
        
        #Calculate unnormalized perturbed internal energy for an ideal gas
        ie = self.delta_p(t, x, y) / (self.gamma - 1)
        
        #Add and normalize
        baseline_TE = 0.5 * self.rho2 * self.U**2
        baseline_TE = baseline_TE + self.p2 / (self.gamma - 1)
        return (ke + ie) / baseline_TE
    
    def delta_TE(self, t, x, y):
        '''
        Total energy perturbation per unit volume 
        in units of [g / (cm * microseconds^2)]
        '''
        #Calculate unnormalized perturbed kinetic energy
        ke = self.delta_KE(t, x, y)
        
        #Calculate unnormalized perturbed internal energy for an ideal gas
        ie = self.delta_p(t, x, y) / (self.gamma - 1)
        
        return ke + ie
    
    ##-----------------------------------------------------------------------
    ## Calculate the energy integrals
    ## Integrals are time dependent and taken over the [0,Lx] x [0,Ly] box in
    ## in the original space coordinates, which is [0,kyLx] x [0,kyLy] in our
    ## dimensionless coordinates
    
    def integral_KE_s(self, t):
        '''
        Sonic-only component of the perturbed kinetic energy integral.
        Returns the (time-dependent) integral of the sonic-only components of
        the perturbed kinetic energy in the fixed spatial box x in [0,Lx] and 
        y in [0,Ly]. Integral is normalized with the area of the box Lx * Ly
        and has units of [g / (cm * microseconds^2)]
        
        IMPORTANT: This function only properly outputs the perturbed integral
        if xs_min > Lx, that is the shock has completely exited the right 
        edge of the box in our dimensionless coordinates
        
        '''
        ## If the shock hasn't completely passed through the region, raise warning
        
        #Calculate minimum x coordinate at the oscillating shock front for time t
        xs_min = self.M2*self.a2 * t - np.abs(np.real(self.delta_xs(t)))
        
        #If xs_min is still in the box region to integrate, raise warning
        #Buffer of 1e-10 is to account for floating point errors when evaluating
        #when the shock front is just leaving the box
        if np.any( xs_min < self.Lx - 1e-10 ):
            #Calculate a t value that would work to push the shock out of the integration region
            if self.longwavelength: 
                if self.Nx == 0:
                    max_delta_xs = self.epsilon_k * self.R * self.M2**2 
                    max_delta_xs = max_delta_xs / (self.ky * math.sqrt(1-self.M2**2))
                else:
                    max_delta_xs = np.abs(np.real(self.delta_xs(self.phi / self.omega)))
            else:
                max_delta_xs = np.abs(np.real(self.delta_xs((pi / 2) * self.omega)))
            t_shock = (self.Lx + max_delta_xs) / (self.M2 * self.a2) 
            
            error = "Shock has not completely passed through the region "
            error = error + "for this choice of t and returned value will be inaccurate. "
            error = error + "For best results, try using t > {t_shock:.4f}".format(t_shock=t_shock)
            warnings.warn(error, Warning)
         
        ## Calculate the x integral part for the long-wavelength condition
        if self.longwavelength:
            
            # Calculate some re-occuring quantities
            eta_r = np.real(self.eta)
            eta_i = np.imag(self.eta)
            norm_eta2 = np.abs(self.eta)**2
            
            exp1 = np.exp( 2 * eta_r * (self.ky*self.Lx - self.M2 * self.ky*self.a2 * t) )
            exp1 = exp1 / (4*self.ky * norm_eta2)
            exp2 = np.exp( -2 * eta_r *self.M2 * self.ky*self.a2 * t)
            exp2 = exp2 / (4*self.ky * norm_eta2)
            
            
            b = - 2*(eta_i*self.ky * self.M2*self.a2 + self.omega) * t
            c = 2*eta_i*self.ky*self.Lx + b
            
            vx0 = self.vx0_s
            vy0 = self.vy0_s
            
            # Build up the x (integral_x) and y (integral_y) components 
            # of the sonic contribution to the integral
            part = exp1 * (norm_eta2 + (eta_r**2)*np.cos(c) + eta_r*eta_i*np.sin(c))
            part = part - exp2 * (norm_eta2 + (eta_r**2)*np.cos(b) + eta_r*eta_i*np.sin(b))
            integral_x = ( np.real(vx0)**2 / eta_r ) * part
            integral_y = ( np.real(vy0)**2 / eta_r ) * part
            
            part = exp1 * (eta_r*np.sin(c) - eta_i*np.cos(c)) 
            part = part + exp2 * (-eta_r*np.sin(b) + eta_i*np.cos(b))
            integral_x = integral_x - 2*np.real(vx0)*np.imag(vx0)*part
            integral_y = integral_y - 2*np.real(vy0)*np.imag(vy0)*part
    
            part = exp1 * (norm_eta2 - (eta_r**2)*np.cos(c) - eta_r*eta_i*np.sin(c))
            part = part + exp2* (-norm_eta2 + (eta_r**2)*np.cos(b) + eta_r*eta_i*np.sin(b))
            integral_x = integral_x + (np.imag(vx0)**2 / eta_r) * part
            integral_y = integral_y + (np.imag(vy0)**2 / eta_r) * part
            
            #Combine x and y components and multiply by coefficient
            integral = integral_x + integral_y
        
        #Calculate the entire x-y integral for the special short-wavelength case where ky = 0
        elif self.Ny == 0:
        
            #Shorthand notation since the integral only involves delta_rho_s and delta_vx_s
            vx0 = self.vx0_s
            rho0 = self.rho0_s
        
            # Calculate some re-occuring quantities
            a = np.imag(self.c)
            b = - (self.omega + a * self.M2*self.a2) * t
            
            #First part of the integral is the 0.5 * rho_2 * delta_vx_s^2 term
            first_integral = 0.5 * self.Lx * (np.abs(vx0)**2)
            
            coeff = 0.5 * (1/a) * np.sin(a*self.Lx) * np.cos(a*self.Lx + 2*b)
            first_integral = first_integral + coeff * (np.real(vx0)**2 - np.imag(vx0)**2)
            
            coeff = (-1/a) * np.sin(a*self.Lx) * np.sin(a*self.Lx + 2*b) 
            first_integral = first_integral + coeff * (np.real(vx0) * np.imag(vx0))
        
            
            #Second part of the integral is the 0.5 * delta_rho_s * delta_vx_s ^2 term
            coeff = np.sin(a*self.Lx + b) * np.cos(2*(a*self.Lx + b)) - np.sin(b) * np.cos(2*b)
            coeff = ( coeff + 5 * (np.sin(a*self.Lx + b) - np.sin(b)) ) / (6*a)
            second_integral = coeff * np.real(rho0) * np.real(vx0)**2
            
            coeff = (np.cos(b)**3 - np.cos(a*self.Lx + b)**3) / (3*a)
            amplitudes = 2*np.real(rho0) * np.real(vx0) * np.imag(vx0)
            amplitudes = amplitudes + np.imag(rho0) * np.real(vx0)**2
            second_integral = second_integral - coeff * amplitudes
            
            coeff = (np.sin(a*self.Lx + b)**3 - np.sin(b)**3) / (3*a)
            amplitudes = 2*np.imag(rho0) * np.real(vx0) * np.imag(vx0)
            amplitudes = amplitudes + np.real(rho0) * np.imag(vx0)**2
            second_integral = second_integral + coeff * amplitudes
            
            coeff = np.cos(3*(a*self.Lx + b)) - np.cos(3*b)
            coeff = ( coeff - 9*(np.cos(a*self.Lx + b) - np.cos(b)) ) / (12*a)
            second_integral = second_integral - coeff * np.imag(rho0) * np.imag(vx0)**2
            
            #Add the first and second terms of the integral
            #Multiply by the y-integral part and the coefficient for the term
            integral = 0.5 * self.rho2 * (self.a2**2) * self.Ly * (first_integral + second_integral)
            return integral
            
        #Otherwise, calculate the x integral part for the short-wavelength condition
        else:
            
            # Calculate some re-occuring quantities
            eta = np.imag(self.eta)
            a = eta * self.ky * self.Lx
            b = a - 2*(eta*self.ky * self.M2*self.a2 + self.omega)* t
            
            vx0 = self.vx0_s
            vy0 = self.vy0_s
            
            # Build up the x (integral_x) and y (integral_y) components 
            # of the sonic contribution to the integral
            integral_x = 0.5 * self.Lx * np.abs(vx0)**2 
            integral_y = 0.5 * self.Lx * np.abs(vy0)**2 
            
            part = np.sin(a) * np.cos(b) / (2*(self.ky)*eta)
            integral_x = integral_x + (np.real(vx0)**2 - np.imag(vx0)**2) * part
            integral_y = integral_y + (np.real(vy0)**2 - np.imag(vy0)**2) * part
            
            part = - np.sin(a) * np.sin(b) / (self.ky*eta)
            integral_x = integral_x + (np.real(vx0) * np.imag(vx0)) * part
            integral_y = integral_y + (np.real(vy0) * np.imag(vy0)) * part

            #Combine x and y components
            integral = integral_x + integral_y
            
        #Multiply by the y-integral part and the coefficient for the term
        integral = 0.5 * self.rho2 * (self.a2**2) * (self.Ly/2) * integral
    
        return integral / (self.Lx * self.Ly)
            
    def integral_KE_v(self):
        '''
        Vortex-only component of the perturbed kinetic energy integral.
        Returns the (time-independent) integral of the vortex-only components of
        the perturbed kinetic energy in the fixed spatial box x in [0,Lx] and 
        y in [0,Ly]. Integral is normalized with the area of the box Lx * Ly
        and has units of [g / (cm * microseconds^2)]
        
        IMPORTANT: This function only properly outputs the perturbed integral
        if xs_min > Lx, that is the shock has completely exited the right 
        edge of the box in our dimensionless coordinates
        '''
        
        #If ky = 0, there is no vortex contribution to the energy
        if self.Ny == 0:
            return 0
        
        #If kx and therefore omega = 0, then we take the limit
        #vx_v = vx0_v and vy_v = vy0_v
        if self.Nx == 0:
            x_integrand = np.real(self.vx0_v)**2 + np.real(self.vy0_v)**2
            x_integral = self.Lx * x_integrand
            return 0.5 * self.rho2 * self.a2**2 * (self.Ly/2) * x_integral
            
        # Calculate some re-occuring quantities
        a = self.omega / (self.M2 * self.a2)
        # a = self.M2 / (self.ky * self.omega)
        # b = self.omega / (self.M2 * self.a2) * self.ky * self.Lx    
        
        vx0 = self.vx0_v
        vy0 = self.vy0_v
            
        # Build up the x (integral_x) and y (integral_y) components of the x-integral
        integral_x = (self.Lx / 2) * np.abs(vx0)**2
        integral_y = (self.Lx / 2) * np.abs(vy0)**2
        
        part = np.sin(2*a*self.Lx) / (4*a)
        # part = (a/4) * np.sin(2*b)
        integral_x = integral_x + (np.real(vx0)**2 - np.imag(vx0)**2) * part
        integral_y = integral_y + (np.real(vy0)**2 - np.imag(vy0)**2) * part
        
        part = (1/a) * np.sin(a*self.Lx)**2
        # part = a * (np.sin(b)**2)
        integral_x = integral_x + np.real(vx0) * np.imag(vx0) * part
        integral_y = integral_y + np.real(vy0) * np.imag(vy0) * part
        
        #Combine x and y components
        integral = integral_x + integral_y
            
        #Multiply by the y-integral part and the coefficient for the term
        integral = 0.5 * self.rho2 * (self.a2**2) * (self.Ly/2) * integral
        
        return integral / (self.Lx * self.Ly)
    
    def integral_KE_mix(self, t):
        '''
        Mixed sonic/vortex cross term component of the perturbed kinetic energy integral.
        Returns the (time-dependent) integral of the mixed components of
        the perturbed kinetic energy in the fixed spatial box x in [0,Lx] and 
        y in [0,Ly]. Integral is normalized with the area of the box Lx * Ly
        and has units of [g / (cm * microseconds^2)]
        
        IMPORTANT: This function only properly outputs the perturbed integral
        if xs_min > Lx, that is the shock has completely exited the right 
        edge of the box in our dimensionless coordinates
        '''
        
        ## If the shock hasn't completely passed through the region, raise warning
        
        #Calculate minimum x coordinate at the oscillating shock front for time t
        xs_min = self.M2*self.a2 * t - np.abs(np.real(self.delta_xs(t)))
        
        #If xs_min is still in the box region to integrate, raise warning
        #Buffer of 1e-10 is to account for floating point errors when evaluating
        #when the shock front is just leaving the box
        if np.any( xs_min < self.Lx - 1e-10 ):
            #Calculate a t value that would work to push the shock out of the integration region
            if self.longwavelength: 
                if self.Nx == 0:
                    max_delta_xs = self.epsilon_k * self.R * self.M2**2 
                    max_delta_xs = max_delta_xs / (self.ky * math.sqrt(1-self.M2**2))
                else:
                    max_delta_xs = np.abs(np.real(self.delta_xs(self.phi / self.omega)))
            else:
                max_delta_xs = np.abs(np.real(self.delta_xs((pi / 2) * self.omega)))
            t_shock = (self.Lx + max_delta_xs) / (self.M2 * self.a2) 
            
            error = "Shock has not completely passed through the region "
            error = error + "for this choice of t and returned value will be inaccurate. "
            error = error + "For best results, try using t > {t_shock:.4f}".format(t_shock=t_shock)
            warnings.warn(error, Warning)
         
            
        ## Shorthand notation for convinience
        vx0_s = self.vx0_s
        vy0_s = self.vy0_s
        vx0_v = self.vx0_v
        vy0_v = self.vy0_v 
        
        ## Calculate the x-integral part for the long-wavelength condition
        if self.longwavelength:
            
            # Calculate some re-occuring quantities
            eta_r = np.real(self.eta)
            eta_i = np.imag(self.eta)
            norm_eta2 = np.abs(self.eta)**2
            
            ap = eta_i + self.omega / (self.ky * self.M2*self.a2) 
            am = eta_i - self.omega / (self.ky * self.M2*self.a2) 
            b = -(eta_i*self.ky * self.M2*self.a2 + self.omega) * t
            zp = ap * self.ky * self.Lx + b
            zm = am * self.ky * self.Lx + b
            
            exp1 = np.exp(eta_r*self.ky * (self.Lx - self.M2*self.a2 * t)) / 2
            exp2 = np.exp(-eta_r*self.ky * self.M2*self.a2 * t) / 2
            cp = norm_eta2 + 2*eta_i * self.omega / (self.ky * self.M2*self.a2) 
            cp = cp + (self.omega / (self.ky * self.M2*self.a2))**2
            cm = norm_eta2 - 2*eta_i * self.omega / (self.ky * self.M2*self.a2) 
            cm = cm + (self.omega / (self.ky * self.M2*self.a2))**2
        
            # Build up the x (integral_x) and y (integral_y) components 
            # of the mixed term contribution to the x-integral
        
            ## Calculate the x-integral part for kx = 0 case
            if self.Nx == 0:
                #There is no vortex component of y-velocity perturbation if kx=0
                #So the y-velocity does not contribute to the mixed energy integral
                integral_y = 0
                
                part = 2*exp1 * (eta_r*np.cos(zp) + eta_i*np.sin(zp))
                part = part - exp2 * (eta_r*np.cos(b) + eta_i*np.sin(b))
                integral_x = np.real(vx0_s) * np.real(vx0_v) * part / norm_eta2
                
                part = 2*exp1 * (eta_r*np.sin(zp) - eta_i*np.cos(zp))
                part = part - exp2 * (eta_r*np.sin(b) - eta_i*np.cos(b))
                integral_x = integral_x - np.imag(vx0_s) * np.real(vx0_v) * part / norm_eta2
            
            # For kx != 0 calculate as follows
            else:
                part = (exp1/cp) * (eta_r*np.cos(zp) + ap*np.sin(zp))
                part = part + (exp1/cm) * (eta_r*np.cos(zm) + am*np.sin(zm))
                part = part - (exp2/cp) * (eta_r*np.cos(b) + ap*np.sin(b))
                part = part - (exp2/cm) * (eta_r*np.cos(b) + am*np.sin(b))
                integral_x = np.real(vx0_s) * np.real(vx0_v) * part
                integral_y = np.real(vy0_s) * np.real(vy0_v) * part
                
                part = (exp1/cp) * (eta_r*np.cos(zp) + ap*np.sin(zp))
                part = part - (exp1/cm) * (eta_r*np.cos(zm) + am*np.sin(zm))
                part = part - (exp2/cp) * (eta_r*np.cos(b) + ap*np.sin(b))
                part = part + (exp2/cm) * (eta_r*np.cos(b) + am*np.sin(b))
                integral_x = integral_x + np.imag(vx0_s) * np.imag(vx0_v) * part
                integral_y = integral_y + np.imag(vy0_s) * np.imag(vy0_v) * part
                
                part = (exp1/cp) * (eta_r*np.sin(zp) - ap*np.cos(zp))
                part = part - (exp1/cm) * (eta_r*np.sin(zm) - am*np.cos(zm))
                part = part - (exp2/cp) * (eta_r*np.sin(b) - ap*np.cos(b))
                part = part + (exp2/cm) * (eta_r*np.sin(b) - am*np.cos(b))
                integral_x = integral_x + np.real(vx0_s) * np.imag(vx0_v) * part
                integral_y = integral_y + np.real(vy0_s) * np.imag(vy0_v) * part
                
                part = -(exp1/cp) * (eta_r*np.sin(zp) - ap*np.cos(zp))
                part = part - (exp1/cm) * (eta_r*np.sin(zm) - am*np.cos(zm))
                part = part + (exp2/cp) * (eta_r*np.sin(b) - ap*np.cos(b))
                part = part + (exp2/cm) * (eta_r*np.sin(b) - am*np.cos(b))
                integral_x = integral_x + np.imag(vx0_s) * np.real(vx0_v) * part
                integral_y = integral_y + np.imag(vy0_s) * np.real(vy0_v) * part
            
            #Combine x and y components
            integral = integral_x + integral_y
            
        #Calculate the entire x-y integral for the special short-wavelength case where ky = 0
        elif self.Ny == 0:
            
            #The only mixed term in this ky=0 case is 0.5 * delta_rho_e * delta_vx_s^2
            ## Shorthand notation for convinience
            vx0 = self.vx0_s
            rho0 = self.rho0_e
            
            # Calculate some re-occuring quantities
            a = np.imag(self.c)
            b = - (self.omega + a * self.M2*self.a2) * t
            d = self.omega / (self.M2 * self.a2)
            cos_plus = ( np.cos((2*a+d)*self.Lx + 2*b) - np.cos(2*b) ) / (4 * (2*a + d))
            cos_minus = ( np.cos((2*a-d)*self.Lx + 2*b) - np.cos(2*b) ) / (4 * (2*a - d))
            sin_plus = ( np.sin((2*a+d)*self.Lx + 2*b) - np.sin(2*b) ) / (4 * (2*a + d))
            sin_minus = ( np.sin((2*a-d)*self.Lx + 2*b) - np.sin(2*b) ) / (4 * (2*a - d))
        
            # Calculate the x-integral
            coeff = sin_plus + sin_minus + 1/(2*d) * np.sin(d * self.Lx)
            integral = coeff * np.real(rho0) * np.real(vx0)**2
            coeff = -cos_plus + cos_minus - 1/(2*d) * (np.cos(d * self.Lx) - 1)
            integral = integral + coeff * np.imag(rho0) * np.real(vx0)**2
            coeff = -sin_plus - sin_minus + 1/(2*d) * np.sin(d * self.Lx)
            integral = integral + coeff * np.real(rho0) * np.imag(vx0)**2
            coeff = cos_plus - cos_minus - 1/(2*d) * (np.cos(d * self.Lx) - 1)
            integral = integral + coeff * np.imag(rho0) * np.imag(vx0)**2
            coeff = cos_plus + cos_minus
            integral = integral + coeff * 2 * np.real(rho0) * np.real(vx0) * np.imag(vx0)
            coeff = sin_plus - sin_minus
            integral = integral + coeff * 2 * np.imag(rho0) * np.real(vx0) * np.imag(vx0)
        
            #Multiply by the y-integral part and the coefficient for the term
            integral = 0.5 * self.rho2 * (self.a2**2) * self.Ly * integral
            return integral
        
        #Calculate the x integral part for the short-wavelength condition
        else:
            
            # Calculate some re-occuring quantities
            eta = np.imag(self.eta)
            b = -(eta*self.ky * self.M2*self.a2 + self.omega) * t
            
            ap = (eta*self.ky + self.omega/(self.M2*self.a2)) * self.Lx / 2
            am = (eta*self.ky - self.omega/(self.M2*self.a2)) * self.Lx / 2
            cp = 1 / ( eta + self.omega/(self.ky * self.M2*self.a2) )
            cm = 1 / ( eta - self.omega/(self.ky * self.M2*self.a2) )
            
            # Build up the x (integral_x) and y (integral_y) components 
            # of the mixed term contribution to the x-integral
            part = cp * np.sin(ap) * np.cos(ap + b)
            part = part + cm * np.sin(am) * np.cos(am + b)
            integral_x = np.real(vx0_s) * np.real(vx0_v) * part
            integral_y = np.real(vy0_s) * np.real(vy0_v) * part

            part = cp * np.sin(ap) * np.cos(ap + b)
            part = part - cm * np.sin(am) * np.cos(am + b)
            integral_x = integral_x + np.imag(vx0_s) * np.imag(vx0_v) * part
            integral_y = integral_y + np.imag(vy0_s) * np.imag(vy0_v) * part
    
            part = cp * np.sin(ap) * np.sin(ap + b)
            part = part - cp * np.sin(am) * np.sin(am + b)
            integral_x = integral_x + np.real(vx0_s) * np.imag(vx0_v) * part 
            integral_y = integral_y + np.real(vy0_s) * np.imag(vy0_v) * part 
            
            part = - cp * np.sin(ap) * np.sin(ap + b)
            part = part - cp * np.sin(am) * np.sin(am + b)
            integral_x = integral_x + np.imag(vx0_s) * np.real(vx0_v) * part 
            integral_y = integral_y + np.imag(vy0_s) * np.real(vy0_v) * part 
    
            #Combine x and y components
            integral = integral_x + integral_y
            
        #Multiply by the y-integral part and the coefficient for the term
        integral = self.rho2 * (self.a2**2) * (self.Ly/2) * (integral / self.ky)
    
        return integral / (self.Lx * self.Ly)
    
    def integral_KE(self, t):
        '''
        Total perturbed kinetic energy integral.
        Returns the (time-dependent) integral of the perturbed kinetic energy 
        in the fixed spatial box x in [0,Lx] and y in [0,Ly]. Integral is 
        normalized with the area of the box Lx * Ly and has units of 
        [g / (cm * microseconds^2)]
        
        IMPORTANT: This function properly outputs the perturbed integral
        if t > ky*Lx / M2, that is the shock has exited the right edge of the box
        in our dimensionless coordinates
        
        '''
        
        return self.integral_KE_s(t) + self.integral_KE_v() + self.integral_KE_mix(t)
    
    def long_time_average_integral_KE(self, component = 'total'):
        '''
    
        Parameters
        ----------
        component : string, optional
            Specify which contribution to the kinetic energy integral to evaluate.
            Accepted values are 'sonic', 'vortex', 'mixed', and 'total'.
            The default is 'total'. 
            
        Returns
        -------
        Returns the long-time average of the perturbed kinetic energy in the 
        fixed spatial box x in [0,Lx] and y in [0,Ly]. The integral is normalized 
        with the area of the box Lx * Ly and has units of [g / (cm * microseconds^2)].
        
        In the long wavelength case, the initial fluctuations decay rapidly 
        and the long-time average value is just the asymptotic steady-state 
        value as t -> infinity. 
        
        In the short wavelength case, the kinetic energy integral is periodic in
        time. We integrate and average over a single period in the original
        dimensional time coordinate, which is the dimensionless time / (ky*a2).
        '''
        
        #The vortex KE integral component is constant in time, so return this constant
        if component == 'vortex':
            return self.integral_KE_v()
        
        #Otherwise, in the long-wavelength case, return the asymptotic value
        elif self.longwavelength:
            #Pick a sufficiently large t to reach asymptotic value
            t = 1000
            if component == 'sonic':
                return self.integral_KE_s(t)
            elif component == 'mixed':
                return self.integral_KE_mix(t)
            else:
                return self.integral_KE(t)
            
        #Finally, in the short wavelength case, return the average over one
        #period in dimensional time
        else:
            #Calculate dimensionl t needed for shock to exit the box and add some buffer
            max_delta_xs = np.abs(np.real(self.delta_xs((pi / 2) * self.omega)))
            t_shock = (self.Lx + max_delta_xs) / (self.M2 * self.a2)
            t_shock = t_shock + 5
            
            #Calculate the period in dimensional time again
            if self.Ny == 0:
                period = np.imag(self.c) * self.M2*self.a2 + self.omega
            else:
                period = np.imag(self.eta)*self.ky * self.M2*self.a2 + self.omega
            period = np.abs(2* pi / period)
            
            #Integrate the kinetic energy space integral over one period in time
            if component == 'sonic':
                integral = integrate.quad(lambda t: self.integral_KE_s(t),
                                          t_shock, t_shock + period)
            elif component == 'mixed':
                integral = integrate.quad(lambda t: self.integral_KE_mix(t),
                                          t_shock, t_shock + period)
            else:
                integral = integrate.quad(lambda t: self.integral_KE(t),
                                          t_shock, t_shock + period)
                
            #Return the average over time, which is the time integral divided
            #by the time interval of one period
            return integral[0] / period
            
    # def vectorize_functions(self):
    #     '''
    #     Vectorize functions with conditionals to accept numpy array input
    #     '''
    #     # self.delta_xs = np.vectorize(self.delta_xs)
    #     # self.delta_rho_rho1 = np.vectorize(self.delta_rho_rho1)
        
    #     # # self.tilde_p = np.vectorize(self.tilde_p)
    #     # self.tilde_vx_s = np.vectorize(self.tilde_vx_s)
    #     # self.tilde_vx_v = np.vectorize(self.tilde_vx_v)
    #     # # self.tilde_vx = np.vectorize(self.tilde_vx)
    #     # self.tilde_vy_s = np.vectorize(self.tilde_vy_s)
    #     # self.tilde_vy_v = np.vectorize(self.tilde_vy_v)
    #     # # self.tilde_vy = np.vectorize(self.tilde_vy)
    #     # self.tilde_rho_s = np.vectorize(self.tilde_rho_s)
    #     # self.tilde_rho_e = np.vectorize(self.tilde_rho_e)
    #     # self.tilde_rho = np.vectorize(self.tilde_rho)
        
    #     self.div_tilde_v = np.vectorize(self.div_tilde_v)
    #     self.curl_tilde_v = np.vectorize(self.curl_tilde_v)
        
    #     self.integral_KE_s = np.vectorize(self.integral_KE_s)
    #     self.integral_KE_v = np.vectorize(self.integral_KE_v)
    #     self.integral_KE_mix = np.vectorize(self.integral_KE_mix)
    #     self.integral_KE = np.vectorize(self.integral_KE)
        
    #     # self.delta_KE = np.vectorize(self.delta_KE)
    #     # self.delta_TE = np.vectorize(self.delta_TE)
        
    #     # self.delta_p = np.vectorize(self.delta_p)
    #     # self.delta_vx_s = np.vectorize(self.delta_vx_s)
    #     # self.delta_vx_v = np.vectorize(self.delta_vx_v)
    #     # self.delta_vx = np.vectorize(self.delta_vx)
    #     # self.delta_vy_s = np.vectorize(self.delta_vy_s)
    #     # self.delta_vy_v = np.vectorize(self.delta_vy_v)
    #     # self.delta_vy = np.vectorize(self.delta_vy)
    #     # self.delta_rho_s = np.vectorize(self.delta_rho_s)
    #     # self.delta_rho_e = np.vectorize(self.delta_rho_e)
    #     # self.delta_rho = np.vectorize(self.delta_rho)
        
# # %% Extra code to check Rankine-Hugoniot Equations 16-19 are satisfied

# # To check, in each of the tilde_p, tilde_vx_s, tilde_vx_v, tilde_vy_s,
# # tilde_vy_v, tilde_rho_s, tilde_rho_v, and tilde_rho functions, the shock
# # front location if statement needs to be replaced to ignore the shock amplitude
# # That is, change:
# #     if x <= self.M2*self.a2  * t:
# #     # if x <= self.M2*self.a2 * t + np.real(self.delta_xs(t,y)):
# # to:
# #     # if x <= self.M2*self.a2  * t:
# #     if x <= self.M2*self.a2 * t + np.real(self.delta_xs(t,y)):


# # At the shockfront at t = 0, the amplitudes alone should satisfy the eqns

# #Change the Nx and Ny values below to check kx=0, ky=0, long and short wavelength
# # Define parameters
# Nx = 1
# Ny = 10
# M1 = 2
# epsilon_k = 0.01
# gamma = (5/3)

# #initialize class
# sim = single_mode_solution(Nx, Ny, M1, epsilon_k, gamma)

# print('Check Amplitudes')

# print('Equation 16')
# LHS = sim.vy0_s + sim.vy0_v
# RHS = sim.M2 * (sim.R-1) * sim.ky * sim.delta_xs(0)
# print('Error/epsilon: ', (LHS-RHS)/epsilon_k)

# print('\nEquation 17')
# LHS = 2*(sim.vx0_s + sim.vx0_v) - (sim.M1**2 + 1) / ((sim.M1**2)*sim.M2) * sim.p0
# RHS = -sim.M2 * (sim.R-1) * sim.epsilon_k
# print('Error/epsilon: ', (LHS-RHS)/sim.epsilon_k)

# print('\nEquation 18')
# LHS = sim.rho0_s + sim.rho0_e - sim.p0 / ((sim.M1**2)*(sim.M2**2))
# RHS = sim.epsilon_k
# print('Error/epsilon: ', (LHS-RHS)/sim.epsilon_k)

# print('\nEquation 19')
# LHS = (2 / sim.a2) * (-1j * sim.omega) * sim.delta_xs(0,0)
# LHS = LHS - (sim.gamma+1)/(2*sim.M2) * sim.p0
# RHS = -sim.M2 * sim.R * sim.epsilon_k
# print('Error/epsilon: ', (LHS-RHS)/sim.epsilon_k)

# ## Log t plot of error in Rankine-Hugoniot conditions Eqns 16-19
# # Error plots should show curves of very small error - i.e. 1e-10

# print("\n\nRankine Hugoniot Error Plot")

# #Arrays to store the absolute errors in eqns 16-19
# err_16 = []
# err_17 = []
# err_18 = []
# err_19 = []

# #Calculate the error magnitude for each equation for various t
# logts = np.linspace(0,8,500)
# for logt in logts:
#     t = 10**logt
#     xs = sim.M2 * sim.a2 * t
#     y = 0.2
    
#     #Equation 16
#     #Fix the sin/cos discrepency when checking. We take the real part of y first
#     if y == None:
#         LHS = sim.tilde_vy(t, xs, y)
#     else:
#         LHS = sim.tilde_vy(t, xs, y=None) * np.cos(sim.ky * y)
#     RHS = sim.M2 * (sim.R-1) * sim.ky * sim.delta_xs(t, y) 
#     err_16.append((LHS-RHS)/sim.epsilon_k)
    
#     #Equation 17
#     LHS = 2*sim.tilde_vx(t, xs, y) 
#     LHS = LHS - (sim.M1**2 + 1) / ((sim.M1**2)*sim.M2) * sim.tilde_p(t, xs, y)
#     RHS = -sim.M2 * (sim.R-1) * sim.delta_rho_rho1(t, xs, y)
#     err_17.append((LHS-RHS)/sim.epsilon_k)
    
#     #Equation 18
#     LHS = sim.tilde_rho(t,xs,y) - sim.tilde_p(t,xs,y) / ((sim.M1**2)*(sim.M2**2))
#     RHS = sim.delta_rho_rho1(t,xs,y)
#     err_18.append((LHS-RHS)/sim.epsilon_k)
    
#     #Equation 19
#     LHS = (2 / sim.a2) * (-1j * sim.omega) * sim.delta_xs(t,y)
#     LHS = LHS - (sim.gamma+1)/(2*sim.M2) * sim.tilde_p(t, xs, y)
#     RHS = -sim.M2 * sim.R * sim.delta_rho_rho1(t,xs,y)
#     err_19.append((LHS-RHS)/sim.epsilon_k)

# # Plot the error    
# errors = [err_16, err_17, err_18, err_19]

# #Real part of error
# plt.figure()

# for i in range(4):
#     plt.plot(logts, [z.real for z in errors[i]], label = "Eqn "+str(16+i))
    
# plt.xlabel('log_10(t)')
# plt.ylabel('Real error/epsilon')
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

# plt.show()

# #Imaginary part of error
# plt.figure()

# for i in range(4):
#     plt.plot(logts, [z.imag for z in errors[i]], label = "Eqn "+str(16+i))
    
# plt.xlabel('log_10(t)')
# plt.ylabel('Imaginary error/epsilon')
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.) 

# plt.show()