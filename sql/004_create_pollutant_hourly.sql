-- Auxiliary pollutant measurements from other stations, used as model inputs.
-- The forecast target (NO2 at station 5177) stays in no2_hourly.
create table pollutant_hourly (
    sensor_id     integer          not null,
    location_id   integer          not null,  -- OpenAQ location
    parameter     text             not null,  -- e.g. 'no2', 'o3'
    observed_at   timestamptz      not null,  -- start of the hour, UTC
    value_ugm3    double precision,
    coverage_pct  real,
    ingested_at   timestamptz      not null default now(),
    primary key (sensor_id, observed_at)
);

alter table pollutant_hourly enable row level security;