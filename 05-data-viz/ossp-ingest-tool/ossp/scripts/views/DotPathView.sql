CREATE VIEW IF NOT EXISTS "DOTPathView" AS 
WITH RECURSIVE generation AS (
    SELECT FirmwareID,
         ParentComponentID,
		 ParentComponent,
		 '"' || ParentComponent || '" -> "' || ChildComponent || '"' as Path,
         ChildComponentID,
		 ChildComponent,
         0 AS generation_number
    FROM ParentChildView
UNION ALL
    SELECT child.FirmwareID,
         child.ParentComponentID,
         child.ParentComponent,
		 g.Path || ' -> "' || child.ChildComponent || '"',
         child.ChildComponentID,
		 child.ChildComponent,
         generation_number+1 AS generation_number
    FROM ParentChildView child
    JOIN generation g
      ON g.ChildComponentID = child.ParentComponentID
 
)
select FirmwareID, Path || ";" as Path from Generation order by ParentComponent ASC
