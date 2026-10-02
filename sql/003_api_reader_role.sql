-- Read-only role for the public API.
-- Its password is set separately in the Supabase SQL editor and never stored in this file:
--   alter role api_reader with password '...';

create role api_reader with login;

grant usage on schema public to api_reader;
grant select on predictions, no2_hourly to api_reader;

-- Row level security is enabled on these tables, so the role also needs read policies
create policy api_reader_read_predictions on predictions
    for select to api_reader using (true);
create policy api_reader_read_no2 on no2_hourly
    for select to api_reader using (true);