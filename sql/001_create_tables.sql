create table no2_hourly (
    sensor_id     integer          not null,
    observed_at   timestamptz      not null,  -- start of the hour, UTC
    value_ugm3    double precision,           -- null when the hour has no value
    coverage_pct  real,
    ingested_at   timestamptz      not null default now(),
    primary key (sensor_id, observed_at)
);

create table weather_hourly (
    location_key           text        not null,  -- e.g. 'station_5177'
    observed_at            timestamptz not null,  -- UTC
    source                 text        not null,  -- 'archive' or 'forecast'
    temperature_2m         real,
    wind_speed_10m         real,
    wind_direction_10m     real,
    relative_humidity_2m   real,
    boundary_layer_height  real,
    ingested_at            timestamptz not null default now(),
    primary key (location_key, observed_at, source)
);

alter table no2_hourly enable row level security;
alter table weather_hourly enable row level security;