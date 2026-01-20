CREATE View SubComponentExport
AS
select ParentComponentID, Parent.Component as ParentComponent, Parent.Version
as ParentComponentVersion,
ChildComponentID, Child.Component as ChildComponent, Child.Version as
ChildComponentVersion,
RootNode, SubComponent.SBOMID, Parent.FirmwareID,
FirmwareID.FirmwareName, FirmwareID.FirmwareVersion,
FirmwareBomver.BomFileName, FirmwareBomVer.BomRef,
FirmwareBomVer.BomSerialNum, FirmwareBomVer.BomTimeStamp
from SubComponent
inner join FirmwareSummary as Parent
on Parent.ComponentID = SubComponent.ParentComponentID and Parent.SBOMID =
SubComponent.SBOMID
inner join FirmwareSummary as Child
on Child.ComponentID = SubComponent.ChildComponentID and Child.SBOMID =
SubComponent.SBOMID
left join FirmwareBomVer ON SubComponent.SBOMID = FirmwareBomVer.SBOMID
inner Join FirmwareID on FirmwareID.FirmwareID = Parent.FirmwareID