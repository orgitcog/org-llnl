CREATE View AssetDataExport
AS
select AssetData.AssetID,
AssetData.Brand,AssetData.Category,AssetData.DataSources,AssetData.DeviceID, AssetData.FirmwareID,
AssetData.FirmwareName as AssetFirmwareName,
AssetData.FirmwareVersion as AssetFirmwareVersion,
AssetData.Model,AssetData.Name,AssetData.Names,AssetData.OS,AssetData.OSVersion,
AssetData.Roles,AssetData.Site,AssetData.Type,
FirmwareId.bomLink as FirmwarebomLink, FirmwareId.Brand as FirmwareBrand,
FirmwareId.Model as FirmwareModel,
FirmwareId.FirmwareName as FirmwareName, FirmwareId.FirmwareVersion as
FirmwareVersion
from AssetData inner Join FirmwareID on AssetData.FirmwareID =
FirmwareID.FirmwareID