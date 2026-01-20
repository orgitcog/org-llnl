(* ::Package:: *)

(* ::Package:: *)
(**)


(* ::Input::Initialization:: *)
Quiet[LibraryUnload[umlib]];
Quiet[LibraryUnload[umlib]];
umlib = LibraryLoad[FileNameJoin[{NotebookDirectory[], "libUM_wll"}]];
remesh3D = LibraryFunctionLoad[umlib, "wll_remesh_3D", {{Real, 2}, {Integer, 2}, {Integer, 2},Real}, "Void"]; 
createMesh2D = LibraryFunctionLoad[umlib, "wll_create_mesh_2D", {{Real, 2}, {Real, 2}, {Real, 2}, Real, Real}, "Void"];
createMesh3D = LibraryFunctionLoad[umlib, "wll_create_mesh_3D", {{Real, 2}, {Real, 2}, {Real, 3}, Real, Real}, "Void"]; 
getMeshVertices = LibraryFunctionLoad[umlib, "wll_get_mesh_vertices", {Integer}, {Real, 2}]; 
getMeshElements = LibraryFunctionLoad[umlib, "wll_get_mesh_elements", {Integer}, {Integer, 2}];
getMeshBoundaryElements = LibraryFunctionLoad[umlib, "wll_get_mesh_boundary_elements", {Integer}, {Integer, 2}];
applyNormalDisplacement = LibraryFunctionLoad[umlib, "wll_apply_normal_displacement", {{Real, 2}, {Integer, 2}, {Integer, 2}, {Real, 1}, Real, Real, {Real, 1}}, {Real, 2}];
fitEdgesToBounds = LibraryFunctionLoad[umlib, "wll_fit_edges_to_bounds", {{Real, 2}, Real, {Real, 1}}, {Real, 2}];


CreateMesh::WrongArgumentType = "expected a list of either StadiumShape[...] xor CapsuleShape[...] in the first argument";
CreateMesh::WrongMeshType = "expected an ElementMesh composed of linear Tetrahedron elements";
CreateMesh::BoundsShapeError = "Dimensions[bounds] must be {2, 2} for 2D, or {2, 3} for 3D";
CreateMesh::CellSizeWarning = "cellsize value likely too coarse: consider using cellsize < Min[r]";

CreateMesh[segments_, bounds_, cellsize_, offsets_] := Module[{heads, segmentData, rmin, vertices, elements},

If[Dimensions[bounds] != {2,2} && Dimensions[bounds != {2, 3}],
  Message[CreateMesh::BoundsShapeError]
  Return[]
];

heads = Tally[Head /@ segments];
If[Length[heads]!= 1 ||(heads[[1,1]] =!= CapsuleShape && heads[[1,1]] =!= StadiumShape), 
  Message[CreateMesh::WrongArgumentType];
  Return[]
];

If[cellsize > Min[#[[2]] & /@ segments],Message[CreateMesh::CellSizeWarning]];

If[heads[[1,1]] === StadiumShape,
segmentData = {#[[1,1,1]],#[[1,1,2]],#[[1,2,1]],#[[1,2,2]],#[[2]]}& /@ segments;
createMesh2D[segmentData, bounds,offsets, cellsize, 0.0];
vertices = getMeshVertices[2];
elements = getMeshElements[2];
,
segmentData = {#[[1,1,1]],#[[1,1,2]],#[[1,1,3]],#[[1,2,1]],#[[1,2,2]],#[[1,2,3]],#[[2]]}& /@ segments;
createMesh3D[segmentData, bounds, offsets, cellsize, 0.0];
vertices = getMeshVertices[3];
elements = getMeshElements[3];
];
{vertices, elements}
]

CreateMesh[segments_, bounds_, cellsize_] := Module[{},
  If[Dimensions[bounds][[2]] == 2, Return[CreateMesh[segments, bounds, cellsize, {{}}]]];
  If[Dimensions[bounds][[2]] == 3, Return[CreateMesh[segments, bounds, cellsize, {{{}}}]]];
]


Remesh[elementMesh_, cellsize_] := Module[{vertices, elements, bdrElements},
If[Head[elementMesh] != ElementMesh,
  Message[CreateMesh::WrongMeshType];
  Return[]
];

vertices = elementMesh["Coordinates"];
elements = elementMesh["MeshElements"][[1,1]];
bdrElements = elementMesh["BoundaryElements"][[1,1]];
If[Dimensions[vertices, 1] != 3 || Dimensions[elements, 1] != 4 || Dimensions[bdrElements, 1] != 3,
  Message[CreateMesh::WrongMeshType];
  Return[]
];

remesh3D[vertices, elements, bdrElements, cellsize];

ToElementMesh[
 "Coordinates" -> getMeshVertices[3],
 "MeshElements" -> {TetrahedronElement[getMeshElements[3]]},
 "BoundaryElements" -> {TriangleElement[Reverse /@ getMeshBoundaryElements[3]]}
]
]
