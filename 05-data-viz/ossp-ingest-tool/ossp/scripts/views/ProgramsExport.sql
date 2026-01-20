CREATE VIEW ProgramsExport
AS
SELECT ProgramId, Programs.AssetID, Name As AssetName, AssetData.Brand,
AssetData.Model, ProgramName, Running, ProgramComponentID,
FirmwareSummary.Component, FirmwareSummary.Version, FirmwareSummary.FirmwareID
from Programs
inner join FirmwareSummary on FirmwareSummary.ComponentID =
Programs.ProgramComponentID inner join AssetData on
AssetData.AssetID=Programs.AssetID
where Export>0