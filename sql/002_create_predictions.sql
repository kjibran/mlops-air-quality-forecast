create table predictions (
    location_key    text        not null,
    issue_time      timestamptz not null,  -- when the forecast was made (UTC)
    horizon         smallint    not null,  -- hours ahead, 1 to 24
    target_time     timestamptz not null,  -- issue_time + horizon
    predicted_ugm3  real        not null,
    model_version   text        not null,  -- registry version that made it, e.g. '2'
    created_at      timestamptz not null default now(),
    primary key (location_key, issue_time, horizon)
);

create index predictions_target_time_idx on predictions (target_time);

alter table predictions enable row level security;