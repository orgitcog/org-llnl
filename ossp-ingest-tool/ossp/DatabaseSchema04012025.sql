BEGIN TRANSACTION;
CREATE TABLE IF NOT EXISTS "AssetData" (
	"OrgId"	INTEGER,
	"AssetID"	INTEGER UNIQUE,
	"Name"	TEXT,
	"Names"	TEXT,
	"MAC"	TEXT,
	"MACNorm"	TEXT,
	"Category"	TEXT,
	"Model"	TEXT,
	"Type"	TEXT,
	"Risk"	TEXT,
	"Brand"	TEXT,
	"Users"	TEXT,
	"Sensor"	TEXT,
	"IPv4Address"	TEXT,
	"IPv4AddressNorm"	TEXT,
	"OS"	TEXT,
	"OSVersion"	TEXT,
	"Location"	TEXT,
	"Boundaries"	TEXT,
	"IPv6Address"	TEXT,
	"IPv6AddressNorm"	TEXT,
	"Site"	TEXT,
	"Suspended"	TEXT,
	"Alerts"	TEXT,
	"SerialNum"	TEXT,
	"DeviceID"	TEXT,
	"DataSources"	TEXT,
	"FirmwareName"	TEXT,
	"FirmwareVersion"	TEXT,
	"FirmwareID"	INTEGER,
	"Roles"	TEXT,
	"AccessSwitch"	TEXT,
	"PurdueLevel"	TEXT,
	"BusinessImpact"	TEXT,
	"FirstSeen"	TEXT,
	"LastSeen"	TEXT,
	"CollectionToolName"	TEXT,
	"CollectionToolVer"	TEXT,
	"CollectionDateTime"	TEXT,
	"CollectionToolType"	TEXT,
	PRIMARY KEY("AssetID" AUTOINCREMENT)
);
CREATE TABLE IF NOT EXISTS "ComponentLicense" (
	"FirmwareID"	INTEGER,
	"ComponentID"	INTEGER,
	"License"	TEXT COLLATE NOCASE,
	"SPDXReferenceNum"	INTEGER,
	PRIMARY KEY("FirmwareID","ComponentID","License")
);
CREATE TABLE IF NOT EXISTS "FirmwareID" (
	"FirmwareID"	INTEGER NOT NULL UNIQUE,
	"bomLink"	TEXT,
	"binLink"	TEXT,
	"Brand"	TEXT,
	"Model"	TEXT,
	"FirmwareName"	TEXT,
	"FirmwareVersion"	TEXT,
	PRIMARY KEY("FirmwareID" AUTOINCREMENT)
);
CREATE TABLE IF NOT EXISTS "FirmwareSummary" (
	"FirmwareID"	INTEGER NOT NULL,
	"ComponentID"	INTEGER,
	"Component"	TEXT,
	"ComponentType"	TEXT,
	"License"	TEXT,
	"Version"	TEXT,
	"Author"	TEXT,
	"Publisher"	TEXT,
	"ProgrammingLang"	TEXT,
	"CPE"	TEXT,
	"PURL"	INTEGER,
	"BomRef"	TEXT,
	"ExternalLicenseRef"	TEXT,
	"ExternalRepoRef"	TEXT,
	"Export"	INTEGER DEFAULT (0),
	"ComponentName"	TEXT,
	"SBOMID"	INTEGER,
	CONSTRAINT "FIRMWARESUMMARY_PK" PRIMARY KEY("ComponentID")
);
CREATE TABLE IF NOT EXISTS "Organization" (
	"OrgId"	INTEGER NOT NULL UNIQUE,
	"Organization"	TEXT,
	"Type"	TEXT,
	"Continent"	TEXT,
	"Country"	TEXT,
	"Area"	TEXT,
	"Customers"	TEXT,
	"Size"	INTEGER,
	"CISector"	TEXT,
	PRIMARY KEY("OrgId" AUTOINCREMENT)
);
CREATE TABLE IF NOT EXISTS "Programs" (
	"ProgramID"	INTEGER NOT NULL,
	"AssetID"	INTEGER NOT NULL,
	"ProgramName"	TEXT,
	"ProgramComponentID"	INTEGER,
	"Running"	TEXT,
	PRIMARY KEY("ProgramID")
);
CREATE TABLE IF NOT EXISTS "spdxlicensealias" (
	"licenseID"	TEXT,
	"alias"	TEXT UNIQUE
);
CREATE TABLE IF NOT EXISTS "spdxlicenses" (
	"reference"	TEXT,
	"isDeprecatedLicenseId"	TEXT,
	"detailsUrl"	TEXT,
	"referenceNumber"	INTEGER UNIQUE,
	"name"	TEXT,
	"licenseId"	TEXT COLLATE NOCASE,
	"isOsiApproved"	TEXT,
	"isFsfLibre"	TEXT,
	"seeAlso"	TEXT,
	PRIMARY KEY("referenceNumber")
);
CREATE TABLE IF NOT EXISTS "table_meta" (
	"table_name"	TEXT NOT NULL COLLATE NOCASE,
	"column_name"	TEXT NOT NULL COLLATE NOCASE,
	"long_name"	TEXT COLLATE NOCASE,
	"description"	TEXT COLLATE NOCASE,
	"research_question"	TEXT COLLATE NOCASE,
	PRIMARY KEY("table_name","column_name")
);
CREATE TABLE IF NOT EXISTS "table_meta_desc" (
	"table_name"	TEXT UNIQUE,
	"table_description"	TEXT,
	"table_purpose"	TEXT
);
CREATE TABLE IF NOT EXISTS "table_schema" (
	"table_name"	TEXT NOT NULL COLLATE NOCASE,
	"column_name"	TEXT NOT NULL COLLATE NOCASE,
	"type"	TEXT COLLATE NOCASE,
	"key"	TEXT COLLATE NOCASE,
	PRIMARY KEY("table_name","column_name")
);
CREATE TABLE IF NOT EXISTS "SubComponent" (
	"ParentComponentID"	INTEGER NOT NULL,
	"ChildComponentID"	INTEGER NOT NULL,
	"RootNode"	INTEGER,
	"SBOMID"	INTEGER
);
CREATE TABLE IF NOT EXISTS "FirmwareBomVer" (
	"SBOMID"	INTEGER NOT NULL,
	"FirmwareID"	INTEGER,
	"BomFileName"	TEXT,
	"BomSerialNum"	TEXT,
	"BomTimeStamp"	TEXT,
	"BomVersion"	TEXT,
	"BomRef"	TEXT,
	"FileName"	TEXT,
	"ComponentVersion"	TEXT,
	"FileType"	TEXT,
	"ToolName"	TEXT,
	"ToolVendor"	TEXT,
	"ToolVersion"	TEXT,
	PRIMARY KEY("SBOMID" AUTOINCREMENT)
);
CREATE INDEX IF NOT EXISTS "FirmwareIDIndex" ON "FirmwareID" (
	"FirmwareID"
);
CREATE INDEX IF NOT EXISTS "OrgAssets" ON "AssetData" (
	"OrgId",
	"AssetID"
);
CREATE INDEX IF NOT EXISTS "spdxlicenses_licenseId_IDX" ON "spdxlicenses" (
	"licenseId"
);
CREATE INDEX IF NOT EXISTS "FirmwareSummaryBomRef" ON "FirmwareSummary" (
	"BomRef"	ASC,
	"Component"	ASC
);
COMMIT;
