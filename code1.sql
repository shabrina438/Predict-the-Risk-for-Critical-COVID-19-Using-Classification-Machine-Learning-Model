%sql
SELECT * FROM COVID_DATA;


%sql
alter table covid_data 
add (
  Percentage_Of_Risk as (round((CONFIRMED/LAB_TEST)*100))
);


%sql
ALTER TABLE covid_data
ADD (
    output AS (
            CASE 
                WHEN (round((CONFIRMED/LAB_TEST)*100)) <= 3 THEN 'NO RISK' 
                WHEN (round((CONFIRMED/LAB_TEST)*100)) > 3 AND (round((CONFIRMED/LAB_TEST)*100)) <= 10 THEN 'LOW RISK'
                WHEN (round((CONFIRMED/LAB_TEST)*100)) > 10 AND (round((CONFIRMED/LAB_TEST)*100)) <= 20 THEN 'MEDIUM RISK'
                WHEN (round((CONFIRMED/LAB_TEST)*100)) > 20 THEN 'HIGH RISK'
            END)
);

%sql
select * from covid_data;

