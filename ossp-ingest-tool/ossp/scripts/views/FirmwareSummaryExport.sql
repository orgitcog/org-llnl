CREATE VIEW FirmwareSummaryExport
AS
Select FirmwareID.FirmwareName, FirmwareID.FirmwareVersion, FirmwareID.Brand,
FirmwareID.Model, FirmwareBomVer.BomFileName, FirmwareBomVer.BomRef as
SBOMBomRef,
FirmwareBomVer.BomSerialNum, FirmwareBomVer.BomTimeStamp,
FirmwareBomVer.BomVersion, FirmwareSummary.FirmwareID,
FirmwareSummary.ComponentID, FirmwareSummary.Component,
FirmwareSummary.ComponentType, FirmwareSummary.License,
FirmwareSummary.Version,
FirmwareSummary.Author, FirmwareSummary.Publisher,
FirmwareSummary.ProgrammingLang,
FirmwareSummary.CPE, FirmwareSummary.PURL,
FirmwareSummary.BomRef, FirmwareSummary.ExternalLicenseRef,
FirmwareSummary.ExternalRepoRef, FirmwareSummary.ComponentName,
FirmwareSummary.SBOMID
from FirmwareSummary inner JOIN firmwareID ON FirmwareSummary.FirmwareID =
FirmwareID.FirmwareID
LEFT join FirmwareBomVer ON FirmwareSummary.SBOMID = FirmwareBomVer.SBOMID
where FirmwareSummary.EXPORT > 0