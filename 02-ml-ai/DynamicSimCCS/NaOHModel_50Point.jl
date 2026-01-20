#NaOH Absorber Model with 10 point shooting method
#Requires DifferentialEquations.jl, NLSolve.jl,
#Thomas Moore 2023

function AbsorberModelNaOH_50Point(;yG0 = 0.16, 
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
        cN2 = (ptot/(R*TG)-cG-wG)
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
        sol2  = IVP_Solution( x[3],  x[4],  x[5],  x[6],  x[7], L/50)
        sol3  = IVP_Solution( x[8],  x[9], x[10], x[11], x[12], L/50)
        sol4  = IVP_Solution(x[13], x[14], x[15], x[16], x[17], L/50)
        sol5  = IVP_Solution(x[18], x[19], x[20], x[21], x[22], L/50)
        sol6  = IVP_Solution(x[23], x[24], x[25], x[26], x[27], L/50)
        sol7  = IVP_Solution(x[28], x[29], x[30], x[31], x[32], L/50)
        sol8  = IVP_Solution(x[33], x[34], x[35], x[36], x[37], L/50)
        sol9  = IVP_Solution(x[38], x[39], x[40], x[41], x[42], L/50)
        sol10 = IVP_Solution(x[43], x[44], x[45], x[46], x[47], L/50)
        sol11 = IVP_Solution(x[48], x[49], x[50], x[51], x[52], L/50)
        sol12 = IVP_Solution(x[53], x[54], x[55], x[56], x[57], L/50)
        sol13 = IVP_Solution(x[58], x[59], x[60], x[61], x[62], L/50)
        sol14 = IVP_Solution(x[63], x[64], x[65], x[66], x[67], L/50)
        sol15 = IVP_Solution(x[68], x[69], x[70], x[71], x[72], L/50)
        sol16 = IVP_Solution(x[73], x[74], x[75], x[76], x[77], L/50)
        sol17 = IVP_Solution(x[78], x[79], x[80], x[81], x[82], L/50)
        sol18 = IVP_Solution(x[83], x[84], x[85], x[86], x[87], L/50)
        sol19 = IVP_Solution(x[88], x[89], x[90], x[91], x[92], L/50)
        sol20 = IVP_Solution(x[93], x[94], x[95], x[96], x[97], L/50)
        sol21 = IVP_Solution(x[98], x[99], x[100], x[101], x[102], L/50)
        sol22 = IVP_Solution(x[103], x[104], x[105], x[106], x[107], L/50)
        sol23 = IVP_Solution(x[108], x[109], x[110], x[111], x[112], L/50)
        sol24 = IVP_Solution(x[113], x[114], x[115], x[116], x[117], L/50)
        sol25 = IVP_Solution(x[118], x[119], x[120], x[121], x[122], L/50)
        sol26 = IVP_Solution(x[123], x[124], x[125], x[126], x[127], L/50)
        sol27 = IVP_Solution(x[128], x[129], x[130], x[131], x[132], L/50)
        sol28 = IVP_Solution(x[133], x[134], x[135], x[136], x[137], L/50)
        sol29 = IVP_Solution(x[138], x[139], x[140], x[141], x[142], L/50)
        sol30 = IVP_Solution(x[143], x[144], x[145], x[146], x[147], L/50)
        sol31 = IVP_Solution(x[148], x[149], x[150], x[151], x[152], L/50)
        sol32 = IVP_Solution(x[153], x[154], x[155], x[156], x[157], L/50)
        sol33 = IVP_Solution(x[158], x[159], x[160], x[161], x[162], L/50)
        sol34 = IVP_Solution(x[163], x[164], x[165], x[166], x[167], L/50)
        sol35 = IVP_Solution(x[168], x[169], x[170], x[171], x[172], L/50)
        sol36 = IVP_Solution(x[173], x[174], x[175], x[176], x[177], L/50)
        sol37 = IVP_Solution(x[178], x[179], x[180], x[181], x[182], L/50)
        sol38 = IVP_Solution(x[183], x[184], x[185], x[186], x[187], L/50)
        sol39 = IVP_Solution(x[188], x[189], x[190], x[191], x[192], L/50)
        sol40 = IVP_Solution(x[193], x[194], x[195], x[196], x[197], L/50)
        sol41 = IVP_Solution(x[198], x[199], x[200], x[201], x[202], L/50)
        sol42 = IVP_Solution(x[203], x[204], x[205], x[206], x[207], L/50)
        sol43 = IVP_Solution(x[208], x[209], x[210], x[211], x[212], L/50)
        sol44 = IVP_Solution(x[213], x[214], x[215], x[216], x[217], L/50)
        sol45 = IVP_Solution(x[218], x[219], x[220], x[221], x[222], L/50)
        sol46 = IVP_Solution(x[223], x[224], x[225], x[226], x[227], L/50)
        sol47 = IVP_Solution(x[228], x[229], x[230], x[231], x[232], L/50)
        sol48 = IVP_Solution(x[233], x[234], x[235], x[236], x[237], L/50)
        sol49 = IVP_Solution(x[238], x[239], x[240], x[241], x[242], L/50)
        sol50 = IVP_Solution(x[243], x[244], x[245], x[246], x[247], L/50)
        if sol1.t[end] < L/50
            resid[1] = 1e5*(L/8 - sol1.t[end])
        elseif sol2.t[end] < L/50
            resid[1] = 1e5*(L/8 - sol2.t[end])
        elseif sol3.t[end] < L/50
            resid[1] = 1e5*(L/8 - sol3.t[end])
        elseif sol4.t[end] < L/50
            resid[1] = 1e5*(L/8 - sol4.t[end])
        elseif sol5.t[end] < L/50
            resid[1] = 1e5*(L/8 - sol5.t[end])
        elseif sol6.t[end] < L/50
            resid[1] = 1e5*(L/8 - sol6.t[end])
        elseif sol7.t[end] < L/50
            resid[1] = 1e5*(L/8 - sol7.t[end])
        elseif sol8.t[end] < L/50
            resid[1] = 1e5*(L/8 - sol8.t[end])
        elseif sol9.t[end] < L/50
            resid[1] = 1e5*(L/8 - sol9.t[end])
        elseif sol10.t[end] < L/50
            resid[1] = 1e5*(L/8 - sol10.t[end])
        elseif sol11.t[end] < L/50
            resid[1] = 1e5*(L/8 - sol11.t[end])
        elseif sol12.t[end] < L/50
            resid[1] = 1e5*(L/8 - sol12.t[end])
        elseif sol13.t[end] < L/50
            resid[1] = 1e5*(L/8 - sol13.t[end])
        elseif sol14.t[end] < L/50
            resid[1] = 1e5*(L/8 - sol14.t[end])
        elseif sol15.t[end] < L/50
            resid[1] = 1e5*(L/8 - sol15.t[end])
        elseif sol16.t[end] < L/50
            resid[1] = 1e5*(L/8 - sol16.t[end])
        elseif sol17.t[end] < L/50
            resid[1] = 1e5*(L/8 - sol17.t[end])
        elseif sol18.t[end] < L/50
            resid[1] = 1e5*(L/8 - sol18.t[end])
        elseif sol19.t[end] < L/50
            resid[1] = 1e5*(L/8 - sol19.t[end])
        elseif sol20.t[end] < L/50
            resid[1] = 1e5*(L/8 - sol20.t[end])
        elseif sol21.t[end] < L/50
            resid[1] = 1e5*(L/8 - sol21.t[end])
        elseif sol22.t[end] < L/50
            resid[1] = 1e5*(L/8 - sol22.t[end])
        elseif sol23.t[end] < L/50
            resid[1] = 1e5*(L/8 - sol23.t[end])
        elseif sol24.t[end] < L/50
            resid[1] = 1e5*(L/8 - sol24.t[end])
        elseif sol25.t[end] < L/50
            resid[1] = 1e5*(L/8 - sol25.t[end])
        elseif sol26.t[end] < L/50
            resid[1] = 1e5*(L/8 - sol26.t[end])
        elseif sol27.t[end] < L/50
            resid[1] = 1e5*(L/8 - sol27.t[end])
        elseif sol28.t[end] < L/50
            resid[1] = 1e5*(L/8 - sol28.t[end])
        elseif sol29.t[end] < L/50
            resid[1] = 1e5*(L/8 - sol29.t[end])
        elseif sol30.t[end] < L/50
            resid[1] = 1e5*(L/8 - sol30.t[end])
        elseif sol31.t[end] < L/50
            resid[1] = 1e5*(L/8 - sol31.t[end])
        elseif sol32.t[end] < L/50
            resid[1] = 1e5*(L/8 - sol32.t[end])
        elseif sol33.t[end] < L/50
            resid[1] = 1e5*(L/8 - sol33.t[end])
        elseif sol34.t[end] < L/50
            resid[1] = 1e5*(L/8 - sol34.t[end])
        elseif sol35.t[end] < L/50
            resid[1] = 1e5*(L/8 - sol35.t[end])
        elseif sol36.t[end] < L/50
            resid[1] = 1e5*(L/8 - sol36.t[end])
        elseif sol37.t[end] < L/50
            resid[1] = 1e5*(L/8 - sol37.t[end])
        elseif sol38.t[end] < L/50
            resid[1] = 1e5*(L/8 - sol38.t[end])
        elseif sol39.t[end] < L/50
            resid[1] = 1e5*(L/8 - sol39.t[end])
        elseif sol40.t[end] < L/50
            resid[1] = 1e5*(L/8 - sol40.t[end])
        elseif sol41.t[end] < L/50
            resid[1] = 1e5*(L/8 - sol41.t[end])
        elseif sol42.t[end] < L/50
            resid[1] = 1e5*(L/8 - sol42.t[end])
        elseif sol43.t[end] < L/50
            resid[1] = 1e5*(L/8 - sol43.t[end])
        elseif sol44.t[end] < L/50
            resid[1] = 1e5*(L/8 - sol44.t[end])
        elseif sol45.t[end] < L/50
            resid[1] = 1e5*(L/8 - sol45.t[end])
        elseif sol46.t[end] < L/50
            resid[1] = 1e5*(L/8 - sol46.t[end])
        elseif sol47.t[end] < L/50
            resid[1] = 1e5*(L/8 - sol47.t[end])
        elseif sol48.t[end] < L/50
            resid[1] = 1e5*(L/8 - sol48.t[end])
        elseif sol49.t[end] < L/50
            resid[1] = 1e5*(L/8 - sol49.t[end])
        elseif sol50.t[end] < L/50
            resid[1] = 1e5*(L/8 - sol50.t[end])
        else
            resid[1]  = sol1(L/50)[1] - x[3]
            resid[2]  = sol1(L/50)[2] - x[4]
            resid[3]  = sol1(L/50)[3] - x[5]
            resid[4]  = sol1(L/50)[4] - x[6]
            resid[5]  = sol1(L/50)[5] - x[7]
            resid[6]  = sol2(L/50)[1] - x[8]
            resid[7]  = sol2(L/50)[2] - x[9]
            resid[8]  = sol2(L/50)[3] - x[10]
            resid[9]  = sol2(L/50)[4] - x[11]
            resid[10] = sol2(L/50)[5] - x[12]
            resid[11] = sol3(L/50)[1] - x[13]
            resid[12] = sol3(L/50)[2] - x[14]
            resid[13] = sol3(L/50)[3] - x[15]
            resid[14] = sol3(L/50)[4] - x[16]
            resid[15] = sol3(L/50)[5] - x[17]
            resid[16] = sol4(L/50)[1] - x[18]
            resid[17] = sol4(L/50)[2] - x[19]
            resid[18] = sol4(L/50)[3] - x[20]
            resid[19] = sol4(L/50)[4] - x[21]
            resid[20] = sol4(L/50)[5] - x[22]
            resid[21] = sol5(L/50)[1] - x[23]
            resid[22] = sol5(L/50)[2] - x[24]
            resid[23] = sol5(L/50)[3] - x[25]
            resid[24] = sol5(L/50)[4] - x[26]
            resid[25] = sol5(L/50)[5] - x[27]
            resid[26] = sol6(L/50)[1] - x[28]
            resid[27] = sol6(L/50)[2] - x[29]
            resid[28] = sol6(L/50)[3] - x[30]
            resid[29] = sol6(L/50)[4] - x[31]
            resid[30] = sol6(L/50)[5] - x[32]
            resid[31] = sol7(L/50)[1] - x[33]
            resid[32] = sol7(L/50)[2] - x[34]
            resid[33] = sol7(L/50)[3] - x[35]
            resid[34] = sol7(L/50)[4] - x[36]
            resid[35] = sol7(L/50)[5] - x[37]
            resid[36] = sol8(L/50)[1] - x[38]
            resid[37] = sol8(L/50)[2] - x[39]
            resid[38] = sol8(L/50)[3] - x[40]
            resid[39] = sol8(L/50)[4] - x[41]
            resid[40] = sol8(L/50)[5] - x[42]
            resid[41] = sol9(L/50)[1] - x[43]
            resid[42] = sol9(L/50)[2] - x[44]
            resid[43] = sol9(L/50)[3] - x[45]
            resid[44] = sol9(L/50)[4] - x[46]
            resid[45] = sol9(L/50)[5] - x[47]
            resid[46] = sol10(L/50)[1] - x[48]
            resid[47] = sol10(L/50)[2] - x[49]
            resid[48] = sol10(L/50)[3] - x[50]
            resid[49] = sol10(L/50)[4] - x[51]
            resid[50] = sol10(L/50)[5] - x[52]
            resid[51] = sol11(L/50)[1] - x[53]
            resid[52] = sol11(L/50)[2] - x[54]
            resid[53] = sol11(L/50)[3] - x[55]
            resid[54] = sol11(L/50)[4] - x[56]
            resid[55] = sol11(L/50)[5] - x[57]
            resid[56] = sol12(L/50)[1] - x[58]
            resid[57] = sol12(L/50)[2] - x[59]
            resid[58] = sol12(L/50)[3] - x[60]
            resid[59] = sol12(L/50)[4] - x[61]
            resid[60] = sol12(L/50)[5] - x[62]
            resid[61] = sol13(L/50)[1] - x[63]
            resid[62] = sol13(L/50)[2] - x[64]
            resid[63] = sol13(L/50)[3] - x[65]
            resid[64] = sol13(L/50)[4] - x[66]
            resid[65] = sol13(L/50)[5] - x[67]
            resid[66] = sol14(L/50)[1] - x[68]
            resid[67] = sol14(L/50)[2] - x[69]
            resid[68] = sol14(L/50)[3] - x[70]
            resid[69] = sol14(L/50)[4] - x[71]
            resid[70] = sol14(L/50)[5] - x[72]
            resid[71] = sol15(L/50)[1] - x[73]
            resid[72] = sol15(L/50)[2] - x[74]
            resid[73] = sol15(L/50)[3] - x[75]
            resid[74] = sol15(L/50)[4] - x[76]
            resid[75] = sol15(L/50)[5] - x[77]
            resid[76] = sol16(L/50)[1] - x[78]
            resid[77] = sol16(L/50)[2] - x[79]
            resid[78] = sol16(L/50)[3] - x[80]
            resid[79] = sol16(L/50)[4] - x[81]
            resid[80] = sol16(L/50)[5] - x[82]
            resid[81] = sol17(L/50)[1] - x[83]
            resid[82] = sol17(L/50)[2] - x[84]
            resid[83] = sol17(L/50)[3] - x[85]
            resid[84] = sol17(L/50)[4] - x[86]
            resid[85] = sol17(L/50)[5] - x[87]
            resid[86] = sol18(L/50)[1] - x[88]
            resid[87] = sol18(L/50)[2] - x[89]
            resid[88] = sol18(L/50)[3] - x[90]
            resid[89] = sol18(L/50)[4] - x[91]
            resid[90] = sol18(L/50)[5] - x[92]
            resid[91] = sol19(L/50)[1] - x[93]
            resid[92] = sol19(L/50)[2] - x[94]
            resid[93] = sol19(L/50)[3] - x[95]
            resid[94] = sol19(L/50)[4] - x[96]
            resid[95] = sol19(L/50)[5] - x[97]
            resid[96] = sol20(L/50)[1] - x[98]
            resid[97] = sol20(L/50)[2] - x[99]
            resid[98] = sol20(L/50)[3] - x[100]
            resid[99] = sol20(L/50)[4] - x[101]
            resid[100] = sol20(L/50)[5] - x[102]
            resid[101] = sol21(L/50)[1] - x[103]
            resid[102] = sol21(L/50)[2] - x[104]
            resid[103] = sol21(L/50)[3] - x[105]
            resid[104] = sol21(L/50)[4] - x[106]
            resid[105] = sol21(L/50)[5] - x[107]
            resid[106] = sol22(L/50)[1] - x[108]
            resid[107] = sol22(L/50)[2] - x[109]
            resid[108] = sol22(L/50)[3] - x[110]
            resid[109] = sol22(L/50)[4] - x[111]
            resid[110] = sol22(L/50)[5] - x[112]
            resid[111] = sol23(L/50)[1] - x[113]
            resid[112] = sol23(L/50)[2] - x[114]
            resid[113] = sol23(L/50)[3] - x[115]
            resid[114] = sol23(L/50)[4] - x[116]
            resid[115] = sol23(L/50)[5] - x[117]
            resid[116] = sol24(L/50)[1] - x[118]
            resid[117] = sol24(L/50)[2] - x[119]
            resid[118] = sol24(L/50)[3] - x[120]
            resid[119] = sol24(L/50)[4] - x[121]
            resid[120] = sol24(L/50)[5] - x[122]
            resid[121] = sol25(L/50)[1] - x[123]
            resid[122] = sol25(L/50)[2] - x[124]
            resid[123] = sol25(L/50)[3] - x[125]
            resid[124] = sol25(L/50)[4] - x[126]
            resid[125] = sol25(L/50)[5] - x[127]
            resid[126] = sol26(L/50)[1] - x[128]
            resid[127] = sol26(L/50)[2] - x[129]
            resid[128] = sol26(L/50)[3] - x[130]
            resid[129] = sol26(L/50)[4] - x[131]
            resid[130] = sol26(L/50)[5] - x[132]
            resid[131] = sol27(L/50)[1] - x[133]
            resid[132] = sol27(L/50)[2] - x[134]
            resid[133] = sol27(L/50)[3] - x[135]
            resid[134] = sol27(L/50)[4] - x[136]
            resid[135] = sol27(L/50)[5] - x[137]
            resid[136] = sol28(L/50)[1] - x[138]
            resid[137] = sol28(L/50)[2] - x[139]
            resid[138] = sol28(L/50)[3] - x[140]
            resid[139] = sol28(L/50)[4] - x[141]
            resid[140] = sol28(L/50)[5] - x[142]
            resid[141] = sol29(L/50)[1] - x[143]
            resid[142] = sol29(L/50)[2] - x[144]
            resid[143] = sol29(L/50)[3] - x[145]
            resid[144] = sol29(L/50)[4] - x[146]
            resid[145] = sol29(L/50)[5] - x[147]
            resid[146] = sol30(L/50)[1] - x[148]
            resid[147] = sol30(L/50)[2] - x[149]
            resid[148] = sol30(L/50)[3] - x[150]
            resid[149] = sol30(L/50)[4] - x[151]
            resid[150] = sol30(L/50)[5] - x[152]
            resid[151] = sol31(L/50)[1] - x[153]
            resid[152] = sol31(L/50)[2] - x[154]
            resid[153] = sol31(L/50)[3] - x[155]
            resid[154] = sol31(L/50)[4] - x[156]
            resid[155] = sol31(L/50)[5] - x[157]
            resid[156] = sol32(L/50)[1] - x[158]
            resid[157] = sol32(L/50)[2] - x[159]
            resid[158] = sol32(L/50)[3] - x[160]
            resid[159] = sol32(L/50)[4] - x[161]
            resid[160] = sol32(L/50)[5] - x[162]
            resid[161] = sol33(L/50)[1] - x[163]
            resid[162] = sol33(L/50)[2] - x[164]
            resid[163] = sol33(L/50)[3] - x[165]
            resid[164] = sol33(L/50)[4] - x[166]
            resid[165] = sol33(L/50)[5] - x[167]
            resid[166] = sol34(L/50)[1] - x[168]
            resid[167] = sol34(L/50)[2] - x[169]
            resid[168] = sol34(L/50)[3] - x[170]
            resid[169] = sol34(L/50)[4] - x[171]
            resid[170] = sol34(L/50)[5] - x[172]
            resid[171] = sol35(L/50)[1] - x[173]
            resid[172] = sol35(L/50)[2] - x[174]
            resid[173] = sol35(L/50)[3] - x[175]
            resid[174] = sol35(L/50)[4] - x[176]
            resid[175] = sol35(L/50)[5] - x[177]
            resid[176] = sol36(L/50)[1] - x[178]
            resid[177] = sol36(L/50)[2] - x[179]
            resid[178] = sol36(L/50)[3] - x[180]
            resid[179] = sol36(L/50)[4] - x[181]
            resid[180] = sol36(L/50)[5] - x[182]
            resid[181] = sol37(L/50)[1] - x[183]
            resid[182] = sol37(L/50)[2] - x[184]
            resid[183] = sol37(L/50)[3] - x[185]
            resid[184] = sol37(L/50)[4] - x[186]
            resid[185] = sol37(L/50)[5] - x[187]
            resid[186] = sol38(L/50)[1] - x[188]
            resid[187] = sol38(L/50)[2] - x[189]
            resid[188] = sol38(L/50)[3] - x[190]
            resid[189] = sol38(L/50)[4] - x[191]
            resid[190] = sol38(L/50)[5] - x[192]
            resid[191] = sol39(L/50)[1] - x[193]
            resid[192] = sol39(L/50)[2] - x[194]
            resid[193] = sol39(L/50)[3] - x[195]
            resid[194] = sol39(L/50)[4] - x[196]
            resid[195] = sol39(L/50)[5] - x[197]
            resid[196] = sol40(L/50)[1] - x[198]
            resid[197] = sol40(L/50)[2] - x[199]
            resid[198] = sol40(L/50)[3] - x[200]
            resid[199] = sol40(L/50)[4] - x[201]
            resid[200] = sol40(L/50)[5] - x[202]
            resid[201] = sol41(L/50)[1] - x[203]
            resid[202] = sol41(L/50)[2] - x[204]
            resid[203] = sol41(L/50)[3] - x[205]
            resid[204] = sol41(L/50)[4] - x[206]
            resid[205] = sol41(L/50)[5] - x[207]
            resid[206] = sol42(L/50)[1] - x[208]
            resid[207] = sol42(L/50)[2] - x[209]
            resid[208] = sol42(L/50)[3] - x[210]
            resid[209] = sol42(L/50)[4] - x[211]
            resid[210] = sol42(L/50)[5] - x[212]
            resid[211] = sol43(L/50)[1] - x[213]
            resid[212] = sol43(L/50)[2] - x[214]
            resid[213] = sol43(L/50)[3] - x[215]
            resid[214] = sol43(L/50)[4] - x[216]
            resid[215] = sol43(L/50)[5] - x[217]
            resid[216] = sol44(L/50)[1] - x[218]
            resid[217] = sol44(L/50)[2] - x[219]
            resid[218] = sol44(L/50)[3] - x[220]
            resid[219] = sol44(L/50)[4] - x[221]
            resid[220] = sol44(L/50)[5] - x[222]
            resid[221] = sol45(L/50)[1] - x[223]
            resid[222] = sol45(L/50)[2] - x[224]
            resid[223] = sol45(L/50)[3] - x[225]
            resid[224] = sol45(L/50)[4] - x[226]
            resid[225] = sol45(L/50)[5] - x[227]
            resid[226] = sol46(L/50)[1] - x[228]
            resid[227] = sol46(L/50)[2] - x[229]
            resid[228] = sol46(L/50)[3] - x[230]
            resid[229] = sol46(L/50)[4] - x[231]
            resid[230] = sol46(L/50)[5] - x[232]
            resid[231] = sol47(L/50)[1] - x[233]
            resid[232] = sol47(L/50)[2] - x[234]
            resid[233] = sol47(L/50)[3] - x[235]
            resid[234] = sol47(L/50)[4] - x[236]
            resid[235] = sol47(L/50)[5] - x[237]
            resid[236] = sol48(L/50)[1] - x[238]
            resid[237] = sol48(L/50)[2] - x[239]
            resid[238] = sol48(L/50)[3] - x[240]
            resid[239] = sol48(L/50)[4] - x[241]
            resid[240] = sol48(L/50)[5] - x[242]
            resid[241] = sol49(L/50)[1] - x[243]
            resid[242] = sol49(L/50)[2] - x[244]
            resid[243] = sol49(L/50)[3] - x[245]
            resid[244] = sol49(L/50)[4] - x[246]
            resid[245] = sol49(L/50)[5] - x[247]
            resid[246] = sol50(L/50)[2] - cL0
            resid[247] = sol50(L/50)[5] - TL0
        end
    end
    
    #------------------#
    #---SOLVE SYSTEM---#
    #------------------#
    x0 =  [cLinit, TL0, 
           cG0, cLinit, wG0, TG0, TL0, cG0, cLinit, wG0, TG0, TL0, cG0, cLinit, wG0, TG0, TL0,
           cG0, cLinit, wG0, TG0, TL0, cG0, cLinit, wG0, TG0, TL0, cG0, cLinit, wG0, TG0, TL0,
           cG0, cLinit, wG0, TG0, TL0, cG0, cLinit, wG0, TG0, TL0, cG0, cLinit, wG0, TG0, TL0,
           cG0, cLinit, wG0, TG0, TL0, cG0, cLinit, wG0, TG0, TL0, cG0, cLinit, wG0, TG0, TL0,
           cG0, cLinit, wG0, TG0, TL0, cG0, cLinit, wG0, TG0, TL0, cG0, cLinit, wG0, TG0, TL0,
           cG0, cLinit, wG0, TG0, TL0, cG0, cLinit, wG0, TG0, TL0, cG0, cLinit, wG0, TG0, TL0,
           cG0, cLinit, wG0, TG0, TL0, cG0, cLinit, wG0, TG0, TL0, cG0, cLinit, wG0, TG0, TL0,
           cG0, cLinit, wG0, TG0, TL0, cG0, cLinit, wG0, TG0, TL0, cG0, cLinit, wG0, TG0, TL0,
           cG0, cLinit, wG0, TG0, TL0, cG0, cLinit, wG0, TG0, TL0, cG0, cLinit, wG0, TG0, TL0,
           cG0, cLinit, wG0, TG0, TL0, cG0, cLinit, wG0, TG0, TL0, cG0, cLinit, wG0, TG0, TL0,
           cG0, cLinit, wG0, TG0, TL0, cG0, cLinit, wG0, TG0, TL0, cG0, cLinit, wG0, TG0, TL0,
           cG0, cLinit, wG0, TG0, TL0, cG0, cLinit, wG0, TG0, TL0, cG0, cLinit, wG0, TG0, TL0,
           cG0, cLinit, wG0, TG0, TL0, cG0, cLinit, wG0, TG0, TL0, cG0, cLinit, wG0, TG0, TL0,
           cG0, cLinit, wG0, TG0, TL0, cG0, cLinit, wG0, TG0, TL0, cG0, cLinit, wG0, TG0, TL0,
           cG0, cLinit, wG0, TG0, TL0, cG0, cLinit, wG0, TG0, TL0, cG0, cLinit, wG0, TG0, TL0,
           cG0, cLinit, wG0, TG0, TL0, cG0, cLinit, wG0, TG0, TL0, cG0, cLinit, wG0, TG0, TL0,
           cG0, cLinit, wG0, TG0, TL0]

    minsol = nlsolve(residuals!, x0, autodiff=:forward, ftol=ftol, iterations = niters)
    min = minsol.zero
    
    #---------------------------#
    #---POST-PROCESS SOLUTION---#
    #---------------------------#
    function sol(z)
        if 0 <= z < L/50
            return IVP_Solution(cG0, min[1], wG0, TG0, min[2], L/50)
        elseif L/50 <= z < 2*L/50
            return IVP_Solution(min[3], min[4], min[5], min[6], min[7], L/50)
        elseif 2L/50 <= z < 3L/50
            return IVP_Solution(min[8], min[9], min[10], min[11], min[12], L/50)
        elseif 3L/50 <= z < 4L/50
            return IVP_Solution(min[13], min[14], min[15], min[16], min[17], L/50)
        elseif 4L/50 <= z < 5L/50
            return IVP_Solution(min[18], min[19], min[20], min[21], min[22], L/50)
        elseif 5L/50 <= z < 6L/50
            return IVP_Solution(min[23], min[24], min[25], min[26], min[27], L/50)
        elseif 6L/50 <= z < 7L/50
            return IVP_Solution(min[28], min[29], min[30], min[31], min[32], L/50)
        elseif 7L/50 <= z < 8L/50
            return IVP_Solution(min[33], min[34], min[35], min[36], min[37], L/50)
        elseif 8L/50 <= z < 9L/50
            return IVP_Solution(min[38], min[39], min[40], min[41], min[42], L/50)
        elseif 9L/50 <= z <= 10L/50
            return IVP_Solution(min[43], min[44], min[45], min[46], min[47], L/50)
        elseif 10L/50 <= z < 11*L/50
            return IVP_Solution(min[48], min[49], min[50], min[51], min[52], L/50)
        elseif 11L/50 <= z < 12*L/50
            return IVP_Solution(min[53], min[54], min[55], min[56], min[57], L/50)
        elseif 12L/50 <= z < 13L/50
            return IVP_Solution(min[58], min[59], min[60], min[61], min[62], L/50)
        elseif 13L/50 <= z < 14L/50
            return IVP_Solution(min[63], min[64], min[65], min[66], min[67], L/50)
        elseif 14L/50 <= z < 15L/50
            return IVP_Solution(min[68], min[69], min[70], min[71], min[72], L/50)
        elseif 15L/50 <= z < 16L/50
            return IVP_Solution(min[73], min[74], min[75], min[76], min[77], L/50)
        elseif 16L/50 <= z < 17L/50
            return IVP_Solution(min[78], min[79], min[80], min[81], min[82], L/50)
        elseif 17L/50 <= z < 18L/50
            return IVP_Solution(min[83], min[84], min[85], min[86], min[87], L/50)
        elseif 18L/50 <= z < 19L/50
            return IVP_Solution(min[88], min[89], min[90], min[91], min[92], L/50)
        elseif 19L/50 <= z <= 20L/50
            return IVP_Solution(min[93], min[94], min[95], min[96], min[97], L/50)
        elseif 20L/50 <= z < 21*L/50
            return IVP_Solution(min[98], min[99], min[100], min[101], min[102], L/50)
        elseif 21L/50 <= z < 22*L/50
            return IVP_Solution(min[103], min[104], min[105], min[106], min[107], L/50)
        elseif 22L/50 <= z < 23L/50
            return IVP_Solution(min[108], min[109], min[110], min[111], min[112], L/50)
        elseif 23L/50 <= z < 24L/50
            return IVP_Solution(min[113], min[114], min[115], min[116], min[117], L/50)
        elseif 24L/50 <= z < 25L/50
            return IVP_Solution(min[118], min[119], min[120], min[121], min[122], L/50)
        elseif 25L/50 <= z < 26L/50
            return IVP_Solution(min[123], min[124], min[125], min[126], min[127], L/50)
        elseif 26L/50 <= z < 27L/50
            return IVP_Solution(min[128], min[129], min[130], min[131], min[132], L/50)
        elseif 27L/50 <= z < 28L/50
            return IVP_Solution(min[133], min[134], min[135], min[136], min[137], L/50)
        elseif 28L/50 <= z < 29L/50
            return IVP_Solution(min[138], min[139], min[140], min[141], min[142], L/50)
        elseif 29L/50 <= z <= 30L/50
            return IVP_Solution(min[143], min[144], min[145], min[146], min[147], L/50)
        elseif 30L/50 <= z < 31*L/50
            return IVP_Solution(min[148], min[149], min[150], min[151], min[152], L/50)
        elseif 31L/50 <= z < 32*L/50
            return IVP_Solution(min[153], min[154], min[155], min[156], min[157], L/50)
        elseif 32L/50 <= z < 33L/50
            return IVP_Solution(min[158], min[159], min[160], min[161], min[162], L/50)
        elseif 33L/50 <= z < 34L/50
            return IVP_Solution(min[163], min[164], min[165], min[166], min[167], L/50)
        elseif 34L/50 <= z < 35L/50
            return IVP_Solution(min[168], min[169], min[170], min[171], min[172], L/50)
        elseif 35L/50 <= z < 36L/50
            return IVP_Solution(min[173], min[174], min[175], min[176], min[177], L/50)
        elseif 36L/50 <= z < 37L/50
            return IVP_Solution(min[178], min[179], min[180], min[181], min[182], L/50)
        elseif 37L/50 <= z < 38L/50
            return IVP_Solution(min[183], min[184], min[185], min[186], min[187], L/50)
        elseif 38L/50 <= z < 39L/50
            return IVP_Solution(min[188], min[189], min[190], min[191], min[192], L/50)
        elseif 39L/50 <= z <= 40L/50
            return IVP_Solution(min[193], min[194], min[195], min[196], min[197], L/50)
        elseif 40L/50 <= z < 41*L/50
            return IVP_Solution(min[198], min[199], min[200], min[201], min[202], L/50)
        elseif 41L/50 <= z < 42*L/50
            return IVP_Solution(min[203], min[204], min[205], min[206], min[207], L/50)
        elseif 42L/50 <= z < 43L/50
            return IVP_Solution(min[208], min[209], min[210], min[211], min[212], L/50)
        elseif 43L/50 <= z < 44L/50
            return IVP_Solution(min[213], min[214], min[215], min[216], min[217], L/50)
        elseif 44L/50 <= z < 45L/50
            return IVP_Solution(min[218], min[219], min[220], min[221], min[222], L/50)
        elseif 45L/50 <= z < 46L/50
            return IVP_Solution(min[223], min[224], min[225], min[226], min[227], L/50)
        elseif 46L/50 <= z < 47L/50
            return IVP_Solution(min[228], min[229], min[230], min[231], min[232], L/50)
        elseif 47L/50 <= z < 48L/50
            return IVP_Solution(min[233], min[234], min[235], min[236], min[237], L/50)
        elseif 48L/50 <= z < 49L/50
            return IVP_Solution(min[238], min[239], min[240], min[241], min[242], L/50)
        elseif 49L/50 <= z <= 50L/50
            return IVP_Solution(min[243], min[244], min[245], min[246], min[247], L/50)
        elseif z > L
            throw("Solution requested at location beyond top of column (i.e. at z > L).")
        end
    end
    
    soldata = zeros(N, 6)
    zvals = LinRange(0, L, N)
    soldata[:, 1] = zvals
    soldata[:, 2] = [sol(z)(mod(z,L/50))[1] for z in zvals]
    soldata[:, 3] = [sol(z)(mod(z,L/50))[2] for z in zvals]
    soldata[:, 4] = [sol(z)(mod(z,L/50))[3] for z in zvals]
    soldata[:, 5] = [sol(z)(mod(z,L/50))[4] for z in zvals]
    soldata[:, 6] = [sol(z)(mod(z,L/50))[5] for z in zvals]
    
    return minsol, sol, soldata
end
