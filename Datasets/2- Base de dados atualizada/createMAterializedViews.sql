/************************ 1. Tickets DATA ************************/
/****** detailed view ******/
IF  EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[MViewTickets]') AND type in (N'U'))
DROP TABLE [dbo].[MViewTickets]
GO
 
SELECT * INTO MViewTickets FROM (
	SELECT	Rostercode, startDateHour, 
			DATEPART(YEAR, Date) as year, DATEPART(MONTH, Date) as month, DATEPART(DAY, Date) as day,
			DATEPART(WEEKDAY, Date) as dayOfWeek, DATEPART(DAYOFYEAR, Date) as dayOfYear, DATEPART(WW, Date) as week,
			DATEPART(HH, StartDateHour) as startHour, DATEPART(MINUTE, StartDateHour) as startMinute,
			CASE WHEN DATEPART(day,StartDateHour) <= 7 THEN CAST('TRUE' AS bit)  
				ELSE CAST('FALSE' AS bit)
			END AS firstOfMonth,
			CASE WHEN DATEPART(MM,StartDateHour) in (1, 3, 5, 7, 8, 10, 12) AND DATEPART(day,StartDateHour) >= 25 THEN CAST('TRUE' AS bit)  
				ELSE CASE WHEN DATEPART(MM,StartDateHour) in (4, 6, 9, 11) AND DATEPART(day,StartDateHour) >= 24 THEN CAST('TRUE' AS bit)
					ELSE CASE WHEN DATEPART(MM,StartDateHour) = 2 AND (  -- tests leap year
								   (DATEPART(year,StartDateHour) % 400) = 0 OR ((DATEPART(year,StartDateHour) % 4) = 0 AND (DATEPART(year,StartDateHour) % 100)<> 0) ) AND
								    DATEPART(day,StartDateHour) >= 23 THEN  CAST('TRUE' AS bit)
						ELSE CASE WHEN  DATEPART(day,StartDateHour) >= 22 THEN  CAST('TRUE' AS bit)  -- non leap year
							ELSE CAST('FALSE' AS bit)
						END
					END
				END
			END AS lastOfMonth,
			CASE WHEN DATEPART(day,StartDateHour) < 8 THEN '1' 
			  ELSE CASE WHEN DATEPART(day,StartDateHour) < 15 then '2' 
				ELSE CASE WHEN  DATEPART(day,StartDateHour) < 22 then '3' 
				  ELSE CASE WHEN  DATEPART(day,StartDateHour) < 29 then '4'     
					ELSE '5'
				  END
				END
			  END
			END AS weekOfMonth,
			0 as holiSpecialDay,
			0 as season,
			CAST(tickets as int) tickets
	FROM	view_Tickets) as A;

ALTER TABLE MViewTickets 
ALTER COLUMN startDateHour datetime NOT NULL;
GO

ALTER TABLE MViewTickets
ADD CONSTRAINT PK_MViewTickets PRIMARY KEY (Rostercode, startDateHour);
GO

CREATE INDEX IDX_MViewTickets_byTimestamp
ON MViewTickets(year, month, day, startHour, startMinute);
GO

CREATE INDEX IDX_MViewTickets_byDayOfWeek
ON MViewTickets(year, month, dayOfWeek, startHour, startMinute);
GO

/****** dayly view ******/
IF  EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[MViewTickets_daily]') AND type in (N'U'))
DROP TABLE [dbo].[MViewTickets_daily]
GO

SELECT * INTO MViewTickets_daily FROM (
	SELECT	Rostercode, CONVERT(DATE, startDateHour) eventDate, year, month, day, dayOfWeek, dayOfYear, week, firstOfMonth, lastOfMonth, weekOfMonth, holiSpecialDay, season, sum(tickets) as tickets
	FROM	MViewTickets
	GROUP BY Rostercode,  CONVERT(DATE, startDateHour), year, month, day, dayOfWeek, dayOfYear, week, firstOfMonth, lastOfMonth, weekOfMonth, holiSpecialDay, season
	) as B
ORDER BY Rostercode, eventDate
GO

ALTER TABLE MViewTickets_daily 
ALTER COLUMN eventDate DATE NOT NULL;
GO

ALTER TABLE MViewTickets_daily
ADD CONSTRAINT PK_MViewTickets_daily PRIMARY KEY (Rostercode, eventDate);
GO

CREATE INDEX IDX_MViewTickets_daily_byDate
ON MViewTickets_daily(year, month, day);
GO

CREATE INDEX IDX_MViewTickets_daily_byDayOfWeek
ON MViewTickets_daily(year, month, dayOfWeek);
GO



/************************ 2. UNITS DATA ************************/
/****** detailed view ******/
IF  EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[MViewUnits]') AND type in (N'U'))
DROP TABLE [dbo].[MViewUnits]
GO
 
SELECT * INTO MViewUnits FROM (
	SELECT	Rostercode, startDateHour, 
			DATEPART(YEAR, Date) as year, DATEPART(MONTH, Date) as month, DATEPART(DAY, Date) as day,
			DATEPART(WEEKDAY, Date) as dayOfWeek, DATEPART(DAYOFYEAR, Date) as dayOfYear,
			DATEPART(HH, StartDateHour) as startHour, DATEPART(MINUTE, StartDateHour) as startMinute,
			CASE WHEN DATEPART(day,StartDateHour) <= 7 THEN CAST('TRUE' AS bit)  
				ELSE CAST('FALSE' AS bit)
			END AS firstOfMonth,
			CASE WHEN DATEPART(MM,StartDateHour) in (1, 3, 5, 7, 8, 10, 12) AND DATEPART(day,StartDateHour) >= 25 THEN CAST('TRUE' AS bit)  
				ELSE CASE WHEN DATEPART(MM,StartDateHour) in (4, 6, 9, 11) AND DATEPART(day,StartDateHour) >= 24 THEN CAST('TRUE' AS bit)
					ELSE CASE WHEN DATEPART(MM,StartDateHour) = 2 AND (  -- tests leap year
								   (DATEPART(year,StartDateHour) % 400) = 0 OR ((DATEPART(year,StartDateHour) % 4) = 0 AND (DATEPART(year,StartDateHour) % 100)<> 0) ) AND
								    DATEPART(day,StartDateHour) >= 23 THEN  CAST('TRUE' AS bit)
						ELSE CASE WHEN  DATEPART(day,StartDateHour) >= 22 THEN  CAST('TRUE' AS bit)  -- non leap year
							ELSE CAST('FALSE' AS bit)
						END
					END
				END
			END AS lastOfMonth,
			CASE WHEN DATEPART(day,StartDateHour) < 8 THEN '1' 
			  ELSE CASE WHEN DATEPART(day,StartDateHour) < 15 then '2' 
				ELSE CASE WHEN  DATEPART(day,StartDateHour) < 22 then '3' 
				  ELSE CASE WHEN  DATEPART(day,StartDateHour) < 29 then '4'     
					ELSE '5'
				  END
				END
			  END
			END AS weekOfMonth,
			0 as holiSpecialDay,
			0 as season,
			units
	FROM	view_Units) as A;
GO

ALTER TABLE MViewUnits 
ALTER COLUMN startDateHour datetime NOT NULL;
GO

ALTER TABLE MViewUnits
ADD CONSTRAINT PK_MViewUnits PRIMARY KEY (Rostercode, startDateHour);
GO

CREATE INDEX IDX_MViewUnits_byTimestamp
ON MViewUnits(year, month, day, startHour, startMinute);
GO

CREATE INDEX IDX_MViewUnits_byDayOfWeek
ON MViewUnits(year, month, dayOfWeek, startHour, startMinute);
GO

/****** dayly view ******/
IF  EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[MViewUnits_daily]') AND type in (N'U'))
DROP TABLE [dbo].[MViewUnits_daily]
GO

SELECT * INTO MViewUnits_daily FROM (
	SELECT	Rostercode, CONVERT(DATE, startDateHour) eventDate, year, month, day, dayOfWeek, dayOfYear, firstOfMonth, lastOfMonth, weekOfMonth, holiSpecialDay, season, sum(Units) as sumUnits
	FROM	MViewUnits
	GROUP BY Rostercode,  CONVERT(DATE, startDateHour), year, month, day, dayOfWeek, dayOfYear, firstOfMonth, lastOfMonth, weekOfMonth, holiSpecialDay, season
	) as B
ORDER BY Rostercode, eventDate
GO

ALTER TABLE MViewUnits_daily 
ALTER COLUMN eventDate DATE NOT NULL;
GO

ALTER TABLE MViewUnits_daily
ADD CONSTRAINT PK_MViewUnits_daily PRIMARY KEY (Rostercode, eventDate);
GO

CREATE INDEX IDX_MViewUnits_daily_byDate
ON MViewUnits_daily(year, month, day);
GO

CREATE INDEX IDX_MViewUnits_daily_byDayOfWeek
ON MViewUnits_daily(year, month, dayOfWeek);
GO

