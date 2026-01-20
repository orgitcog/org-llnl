CREATE VIEW IF NOT EXISTS "GenerationsView" AS 
WITH RECURSIVE generation AS (
    SELECT FirmwareID,
         ParentComponentID,
		 ParentComponent,
         ChildComponentID,
		 ChildComponent,
         0 AS generation_number
    FROM ParentChildView
UNION ALL
    SELECT child.FirmwareID,
         child.ParentComponentID,
         child.ParentComponent,
         child.ChildComponentID,
		 child.ChildComponent,
         generation_number+1 AS generation_number
    FROM ParentChildView child
    JOIN generation g
      ON g.ChildComponentID = child.ParentComponentID
 
)
select * from Generation order by ParentComponent ASC
