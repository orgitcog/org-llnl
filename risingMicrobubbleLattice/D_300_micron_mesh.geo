SetFactory("OpenCASCADE");

scalefactor = 1000;
Mesh.ScalingFactor = 1/scalefactor;


h = .00003*scalefactor;
Mesh.CharacteristicLengthMin = .2*h;
Mesh.CharacteristicLengthMax = 5*h;

L = .0005*scalefactor;//1;
R = .00015*scalefactor;//.25;

numcells = 3;
offset = (numcells+1)*(numcells+1);

For i In {0:numcells}
    For j In {0:numcells}
        
        Cylinder((numcells+1)*i + j + 1) =              {i*L,   j*L,  0,  0,  0,  numcells*L,    R};    
        
        Cylinder((numcells+1)*i + j + 1 + offset) =     {i*L,   0,  j*L,  0,  numcells*L,  0,    R};    

        Cylinder((numcells+1)*i + j + 1 + 2*offset) =   {0,   i*L,  j*L,  numcells*L,  0,  0,    R};    
        // Printf("t= %g",3*i + j + 1);
        // Printf("t= %g",offset);

    EndFor
EndFor


BooleanUnion(49) = { Volume{1}; Delete;}{ Volume{2:48}; Delete;};

Box(50) = {0.5*L,-1.5*L,0.5*L,(numcells-1)*L,numcells*L+1.*L+1.5*L,(numcells-1)*L};

BooleanDifference(51) = { Volume{50}; Delete;}{ Volume{49}; Delete;};

Physical Volume("internal") = {51};

Field[1] = Box;
Field[1].VIn = h/3;
Field[1].VOut = h;

Field[1].XMin = L-.1*L;
Field[1].XMax = 2*L+.1*L;

Field[1].YMin = -10*L;
Field[1].YMax = 10*L;

Field[1].ZMin = L-.1*L;
Field[1].ZMax = 2*L+.1*L;

Field[1].Thickness = L/4;


Field[2] = Min;
Field[2].FieldsList = {1};
Background Field = 2;



array_lowerWall = {};
array_topWall = {};
array_wall = {};
array_sideX = {};
array_sideZ = {};

idx_lowerWall = 0;
idx_topWall = 0;
idx_wall = 0;
idx_sideX = 0;
idx_sideZ = 0;

For i In {1:66}
    //lower wall
    If(i == 2)
        array_lowerWall[idx_lowerWall] = i;
        idx_lowerWall = idx_lowerWall + 1;

    //top wall
    ElseIf(i==5)
        array_topWall[idx_topWall] = i;
        idx_topWall = idx_topWall + 1;

    //side wall X
    ElseIf(i==1 || i==14) 
        array_sideX[idx_sideX] = i;
        idx_sideX = idx_sideX + 1;

    //side wall Z
    ElseIf(i==3 || i==4)
        array_sideZ[idx_sideZ] = i;
        idx_sideZ = idx_sideZ + 1;    

    //all other walls
    Else
        array_wall[idx_wall] = i;
        idx_wall =idx_wall+1;

    EndIf
EndFor

Physical Surface("lowerWall") = {array_lowerWall[]};
Physical Surface("topWall") = {array_topWall[]};
Physical Surface("wall") = {array_wall[]};
Physical Surface("sideX") = {array_sideX[]};
Physical Surface("sideZ") = {array_sideZ[]};
