#NaOH Absorber Model with 10 point shooting method
#Requires DifferentialEquations.jl, NLSolve.jl,
#Thomas Moore 2023

function AbsorberModelNaOH(;
        yG0 = 0.16, 
        RelHumid = 0.5,#
        ptot = 1e5,    #Pa
        cL0 = 0.0,     #mol/m3
        cOH0 = 2000,   #mol/m3
        vG = 0.5,      #m/s
        vL = 0.01,     #m/s
        TG0 = 298,     #K
        TL0 = 298,     #K
        L = 0.5,       #m
        at = 250,      #m2/m3
        σc = 0.04,     #N/m
        σ = 0.055,     #N/m
        μL = 1e-3,     #Pa.s
        ρLM = 1080,    #kg/m3
        cPL = 4000,    #J/kg.K
        R = 8.314,     #J/K.mol
        g = 9.81,      #m/s^2
        MWN2 = 0.028,  #kg/mol
        MWCO2 = 0.044, #kg/mol
        MWH2O = 0.018, #kg/mol
        μG = 1.7e-5,   #Pa.s
        cPG = 1040,    #J/kg.K
        DWG = 2.8e-5,  #m2/s
        atDp = 3.4,    #
        κG = 0.027,    #W/mK
        ΔH = -187000,  #J/mol
        ΔHw = -48000,  #J/mol
        cLinit = 1e2,  #mol/m3
        abstolode = 1e-9,
        reltolode = 1e-9,
        ftol = 1e-8,
        niters = 1000,
        N = 100,
        a_method = "Onda"
    )
    
    
    ################################
    #########DEFINE FUNCTIONS#######
    ################################
    
    #Henry's Constant (mol/m3.Pa)
    #----------------------------
    function H(TL, cL; cOH0 = cOH0, hNa = 0.114e-3, hOH = 0.0839e-3, hCO32 = 0.1423e-3,
                hCO20 = -0.0172e-3, hTCO2 = -0.338e-6)

        HCO2w = 3.54e-7*exp(2044/TL)           #mol/Pa.m3
        cNa = cOH0
        cOH = cOH0 - 2cL
        cCO32 = cL
        return HCO2w/(10^((hNa+hCO20+hTCO2*(TL-298.15))*cNa+
                       (hOH+hCO20+hTCO2*(TL-298.15))*cOH+
                     (hCO32+hCO20+hTCO2*(TL-298.15))*cCO32))
    end
    
    #Diffusivity CO2, m2/s
    #---------------------
    DCO2w(TL) = 2.35e-6*exp(-2119/TL)
    DCO2(TL; cNa = cOH0) = DCO2w(TL)*(1-0.129e-3*cNa)
    
    #Diffusivity OH, m2/s
    #--------------------
    DOH(TL) = 5.27e-9*TL/298.15
    
    #Water vapor pressure, Pa
    #------------------------
    pwvap(TL; xw = 0.9) = 1e5*xw*10^(4.6543 − (1435.264 / (TL - 64.848)))
    
    #Wetted Area, Onda Correlation
    #-----------------------------
    function a(; at=at, σ=σ, σc=σc, ρLM = ρLM, vL=vL, g=g, μL=μL, a_method = a_method)
        if a_method == "Onda"
            return at*(1 - exp(-1.45 * (σc/σ)^0.75 * (ρLM*vL/(at*μL))^0.1 * (vL^2*at/g)^-0.05 * (ρLM*vL^2/(σ*at))^0.2))
        else
            return at
        end
    end
    
    #Liquid Film mass Transfer Coefficient, m/s
    #------------------------------------------
    function kL(TL; at=at, σ=σ, σc=σc, ρLM = ρLM, vL=vL, g=g, μL=μL)
        aval = a(at=at, σ=σ, σc=σc, ρLM=ρLM, vL=vL, g=g, μL=μL)
        return 0.0051 * (vL*ρLM/(aval*μL))^(2/3) * (μL/(ρLM*DCO2(TL)))^(-1/2)*(3.4)^(0.4) * (μL*g/ρLM)^(1/3)
    end
    
    #Gas Film Mass Transfer Coefficient, m/s
    #---------------------------------------
    function kG(TG, cG, wG; R=R, vG=vG, ptot=ptot, MWN2 = MWN2, MWCO2 = MWCO2, MWH2O = MWH2O, 
                μG = μG, atDp = atDp, at=at, DG=DWG)
        cN2 = (ptot/(R*TG)-cG-wG)
        ρGM = cG*MWCO2 + wG*MWH2O + cN2*MWN2
        ReG = vG*ρGM/(at*μG)
        ScG = μG/(ρGM*DG)
        ShG = 2 * ReG^(0.7) * ScG^(1/3) * atDp^(-2)
        return ShG*at*DG/(R*TG)
    end
    
    #Gas Film Heat Transfer Coefficient, m/s
    #---------------------------------------
     function h(TG, cG, wG; R=R, vG=vG, ptot=ptot, MWN2=MWN2, MWCO2=MWCO2, MWH2O=MWH2O, 
                μG = μG, atDp = atDp, at=at, cPG=cPG, κG=κG)
        cN2 = (ptot/(R*TG)-cG-wG)
        ρGM = cG*MWCO2 + wG*MWH2O + cN2*MWN2
        ReG = vG*ρGM/(at*μG)
        PrG = cPG*μG/κG
        NuG = 2 * ReG^(0.7) * PrG^(1/3) * atDp^(-2)
        return NuG*at*κG
    end
    
    #Second order rate constant, mol/m3.s
    #------------------------------------
    function k2(TL, cL; cOH0 = cOH0)
        cNa = cOH0
        cOH = cOH0 - 2cL
        cCO32 = cL
        I = 0.5*(cNa + cOH + 4*cCO32)
        return 1e-3*exp(31.3957 - 6665.9912/TL + 0.0001842*I)
    end
    
    #Enhancement Factor
    #------------------
    function E(TL, cL, TG, cG; cOH0=cOH0, ptot=ptot, R=R)
        pCO2 = cG*R*TG
        cOH = max(0, cOH0 - 2cL)
        Ha = max(0, DCO2(TL)*k2(TL, cL)*cOH/kL(TL)^2)
        Ei = max(1,1 + DOH(TL)*cOH/(2*DCO2(TL)*pCO2*H(TL, cL)))
        E1 = max(1, sqrt(Ha) / tanh(sqrt(Ha)))
        return 1 + (1/((Ei-1)^(-1.35) + (E1-1)^(-1.35)))^(1/1.35)
    end
    
    ####################################
    #####IMPLEMENT SHOOTING METHOD######
    ####################################
    
    #-----------------#
    #---DEFINE ODES---#
    #-----------------#
    function dudz!(dudz,u,p,z)
        
        #Preliminary Calculations
        (cG, cL, wG, TG, TL) = u
        cN2 = ptot/(R*TG)-cG-wG
        ρGM = cG*MWCO2 + wG*MWH2O + cN2*MWN2
        wGstar = pwvap(TL)/(R*TG)
        
        #ODEs
        dudz[1] =  (-E(TL,cL,TG,cG)*kL(TL)*a()*H(TL,cL)*R*TG*cG) / vG 
        dudz[2] = (-E(TL,cL,TG,cG)*kL(TL)*a()*H(TL,cL)*R*TG*cG) / vL
        dudz[3] = (-kG(TG,cG,wG)*a()*R*TG*(wG-wGstar)) / vG
        dudz[4] = (-h(TG,cG,wG)*a()*(TG-TL)) / (vG*cPG*ρGM)
        dudz[5] = (-h(TG,cG,wG)*a()*(TG-TL) + 
                  E(TL,cL,TG,cG)*kL(TL)*a()*H(TL,cL)*R*TG*cG*ΔH + 
                  +kG(TG,cG,wG)*a()*R*TG*(wG-wGstar)*ΔHw) / (vL*cPL*ρLM)
    end
    
    #-------------------------------------------#
    #---FUNCTION TO SOLVE ODES IN ONE SEGMENT---#
    #-------------------------------------------#
    function IVP_Solution(cG, cL, wG, TG, TL, segmentlength)
        u0 = [cG, cL, wG, TG, TL]
        zspan = (0.0, segmentlength)
        IVPprob = ODEProblem(dudz!, u0, zspan)
        return solve(IVPprob, Tsit5(), verbose=false, reltol = reltolode, abstol=abstolode)
    end
    
    #---------------------------------------#
    #---RESIDUAL FUNCTION FOR BVP PROBLEM---#
    #---------------------------------------#
    wG0 = RelHumid*pwvap(TG0,xw=1.0)/(R*TG0)
    cG0 = yG0*ptot/(R*TG0)
    function residuals!(resid, x)
        sol1  = IVP_Solution(cG0,    x[1], wG0, TG0, x[2], L/10)
        sol2  = IVP_Solution( x[3],  x[4],  x[5],  x[6],  x[7], L/10)
        sol3  = IVP_Solution( x[8],  x[9], x[10], x[11], x[12], L/10)
        sol4  = IVP_Solution(x[13], x[14], x[15], x[16], x[17], L/10)
        sol5  = IVP_Solution(x[18], x[19], x[20], x[21], x[22], L/10)
        sol6  = IVP_Solution(x[23], x[24], x[25], x[26], x[27], L/10)
        sol7  = IVP_Solution(x[28], x[29], x[30], x[31], x[32], L/10)
        sol8  = IVP_Solution(x[33], x[34], x[35], x[36], x[37], L/10)
        sol9  = IVP_Solution(x[38], x[39], x[40], x[41], x[42], L/10)
        sol10 = IVP_Solution(x[43], x[44], x[45], x[46], x[47], L/10)
        if sol1.t[end] < L/10
            resid[1] = 1e5*(L/8 - sol1.t[end])
        elseif sol2.t[end] < L/10
            resid[1] = 1e5*(L/8 - sol2.t[end])
        elseif sol3.t[end] < L/10
            resid[1] = 1e5*(L/8 - sol3.t[end])
        elseif sol4.t[end] < L/10
            resid[1] = 1e5*(L/8 - sol4.t[end])
        elseif sol5.t[end] < L/10
            resid[1] = 1e5*(L/8 - sol5.t[end])
        elseif sol6.t[end] < L/10
            resid[1] = 1e5*(L/8 - sol6.t[end])
        elseif sol7.t[end] < L/10
            resid[1] = 1e5*(L/8 - sol7.t[end])
        elseif sol8.t[end] < L/10
            resid[1] = 1e5*(L/8 - sol8.t[end])
        elseif sol9.t[end] < L/10
            resid[1] = 1e5*(L/8 - sol9.t[end])
        elseif sol10.t[end] < L/10
            resid[1] = 1e5*(L/8 - sol10.t[end])
        else
            resid[1]  = sol1(L/10)[1] - x[3]
            resid[2]  = sol1(L/10)[2] - x[4]
            resid[3]  = sol1(L/10)[3] - x[5]
            resid[4]  = sol1(L/10)[4] - x[6]
            resid[5]  = sol1(L/10)[5] - x[7]
            resid[6]  = sol2(L/10)[1] - x[8]
            resid[7]  = sol2(L/10)[2] - x[9]
            resid[8]  = sol2(L/10)[3] - x[10]
            resid[9]  = sol2(L/10)[4] - x[11]
            resid[10] = sol2(L/10)[5] - x[12]
            resid[11] = sol3(L/10)[1] - x[13]
            resid[12] = sol3(L/10)[2] - x[14]
            resid[13] = sol3(L/10)[3] - x[15]
            resid[14] = sol3(L/10)[4] - x[16]
            resid[15] = sol3(L/10)[5] - x[17]
            resid[16] = sol4(L/10)[1] - x[18]
            resid[17] = sol4(L/10)[2] - x[19]
            resid[18] = sol4(L/10)[3] - x[20]
            resid[19] = sol4(L/10)[4] - x[21]
            resid[20] = sol4(L/10)[5] - x[22]
            resid[21] = sol5(L/10)[1] - x[23]
            resid[22] = sol5(L/10)[2] - x[24]
            resid[23] = sol5(L/10)[3] - x[25]
            resid[24] = sol5(L/10)[4] - x[26]
            resid[25] = sol5(L/10)[5] - x[27]
            resid[26] = sol6(L/10)[1] - x[28]
            resid[27] = sol6(L/10)[2] - x[29]
            resid[28] = sol6(L/10)[3] - x[30]
            resid[29] = sol6(L/10)[4] - x[31]
            resid[30] = sol6(L/10)[5] - x[32]
            resid[31] = sol7(L/10)[1] - x[33]
            resid[32] = sol7(L/10)[2] - x[34]
            resid[33] = sol7(L/10)[3] - x[35]
            resid[34] = sol7(L/10)[4] - x[36]
            resid[35] = sol7(L/10)[5] - x[37]
            resid[36] = sol8(L/10)[1] - x[38]
            resid[37] = sol8(L/10)[2] - x[39]
            resid[38] = sol8(L/10)[3] - x[40]
            resid[39] = sol8(L/10)[4] - x[41]
            resid[40] = sol8(L/10)[5] - x[42]
            resid[41] = sol9(L/10)[1] - x[43]
            resid[42] = sol9(L/10)[2] - x[44]
            resid[43] = sol9(L/10)[3] - x[45]
            resid[44] = sol9(L/10)[4] - x[46]
            resid[45] = sol9(L/10)[5] - x[47]
            resid[46] = sol10(L/10)[2]- cL0
            resid[47] = sol10(L/10)[5]- TL0
        end
    end
    
    #------------------#
    #---SOLVE SYSTEM---#
    #------------------#
    x0 =  [cLinit, TL0, 
           cG0, cLinit, wG0, TG0, TL0, cG0, cLinit, wG0, TG0, TL0, cG0, cLinit, wG0, TG0, TL0,
           cG0, cLinit, wG0, TG0, TL0, cG0, cLinit, wG0, TG0, TL0, cG0, cLinit, wG0, TG0, TL0,
           cG0, cLinit, wG0, TG0, TL0, cG0, cLinit, wG0, TG0, TL0, cG0, cLinit, wG0, TG0, TL0]
    minsol = nlsolve(residuals!, x0, autodiff=:forward, ftol=ftol, iterations = niters)
    min = minsol.zero
    
    #---------------------------#
    #---POST-PROCESS SOLUTION---#
    #---------------------------#
    function sol(z)
        if 0 <= z < L/10
            return IVP_Solution(cG0, min[1], wG0, TG0, min[2], L/10)
        elseif L/10 <= z < 2*L/10
            return IVP_Solution(min[3], min[4], min[5], min[6], min[7], L/10)
        elseif 2L/10 <= z < 3L/10
            return IVP_Solution(min[8], min[9], min[10], min[11], min[12], L/10)
        elseif 3L/10 <= z < 4L/10
            return IVP_Solution(min[13], min[14], min[15], min[16], min[17], L/10)
        elseif 4L/10 <= z < 5L/10
            return IVP_Solution(min[18], min[19], min[20], min[21], min[22], L/10)
        elseif 5L/10 <= z < 6L/10
            return IVP_Solution(min[23], min[24], min[25], min[26], min[27], L/10)
        elseif 6L/10 <= z < 7L/10
            return IVP_Solution(min[28], min[29], min[30], min[31], min[32], L/10)
        elseif 7L/10 <= z < 8L/10
            return IVP_Solution(min[33], min[34], min[35], min[36], min[37], L/10)
        elseif 8L/10 <= z < 9L/10
            return IVP_Solution(min[38], min[39], min[40], min[41], min[42], L/10)
        elseif 9L/10 <= z <= L
            return IVP_Solution(min[43], min[44], min[45], min[46], min[47], L/10)
        elseif z > L
            throw("Solution requested at location beyond top of column (i.e. at z > L).")
        end
    end
    
    soldata = zeros(N, 6)
    zvals = LinRange(0, L, N)
    soldata[:, 1] = zvals
    soldata[:, 2] = [sol(z)(mod(z,L/10))[1] for z in zvals]
    soldata[:, 3] = [sol(z)(mod(z,L/10))[2] for z in zvals]
    soldata[:, 4] = [sol(z)(mod(z,L/10))[3] for z in zvals]
    soldata[:, 5] = [sol(z)(mod(z,L/10))[4] for z in zvals]
    soldata[:, 6] = [sol(z)(mod(z,L/10))[5] for z in zvals]
    
    return minsol, sol, soldata
end